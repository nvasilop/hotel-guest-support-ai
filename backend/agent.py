"""
agent.py — Agent flow for StayFlow AI, the AegeanStay Hotels Guest Support
Copilot.

The flow is mostly rule-based (simple and easy to explain), and now connects to
the RAG layer and Gemini when it decides to actually ANSWER a guest question:

High-level flow (see run_agent):
  detect language -> classify intent -> check sensitive/escalation/ambiguity
  -> if the decision is "answer": retrieve relevant knowledge base chunks and
     ask Gemini to write a grounded answer using ONLY those chunks.

For clarify / fallback / escalate decisions we do NOT call RAG or Gemini — we
just return a ready-made template from the playbook.
"""

import os
import re

from dotenv import load_dotenv
from google import genai

from playbook import SYSTEM_PROMPT, get_template
from rag import retrieve

# Load environment variables (e.g. GEMINI_API_KEY) from a .env file if present.
load_dotenv()

# A simple, fast Gemini model used to generate the grounded answers.
GENERATION_MODEL = "gemini-2.5-flash"

# In-memory cache of the Gemini client so we create it only once.
_CLIENT = None


def _get_client() -> genai.Client:
    """Create (once) and return the Gemini client.

    Raises a clear error if GEMINI_API_KEY is not set.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to backend/.env "
                "(see backend/.env.example) before generating answers."
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


# --- Language detection -----------------------------------------------------


def detect_language(message: str) -> str:
    """Return "el" if the message contains Greek characters, else "en"."""
    for char in message:
        # Greek and Greek Extended Unicode ranges.
        if "\u0370" <= char <= "\u03ff" or "\u1f00" <= char <= "\u1fff":
            return "el"
    return "en"


# --- Intent classification --------------------------------------------------
# Simple keyword lists per hotel guest support intent. We lowercase the message
# and check if any keyword appears in it. The order of checks below sets priority.

BOOKING_KEYWORDS = [
    "booking", "book a room", "reservation", "reserve", "availability", "rooms",
    "κράτηση", "κρατήσεις", "διαθεσιμότητα", "δωμάτιο",
]

CHECKIN_CHECKOUT_KEYWORDS = [
    "check-in", "check in", "checkin", "checkout", "check-out", "check out",
    "late checkout", "early check-in", "arrival", "departure",
    "άφιξη", "αναχώρηση", "τσεκ ιν", "τσεκ άουτ",
]

CANCELLATION_REFUND_KEYWORDS = [
    "cancel", "cancel booking", "cancel my booking", "cancel reservation",
    "cancellation", "booking cancellation", "refund", "no-show", "no show",
    "ακύρωση", "ακύρωση κράτησης", "ακυρώσω", "μπορώ να ακυρώσω",
    "επιστροφή χρημάτων", "επιστροφή χρημ",
]

AMENITIES_KEYWORDS = [
    "breakfast", "pool", "spa", "parking", "wi-fi", "wifi", "gym",
    "room service", "beach", "towels",
    "πρωινό", "πισίνα", "πάρκινγκ", "παραλία", "γυμναστήριο",
]

PAYMENTS_KEYWORDS = [
    "payment", "pay", "card", "invoice", "receipt", "deposit", "charged",
    "charge", "billing",
    "χρέωση", "πληρωμή", "απόδειξη", "τιμολόγιο", "προκαταβολή",
]

# Room issue detection is a bit smarter than a flat keyword list, because a
# generic word like "problem"/"πρόβλημα" is only a room issue when it is about a
# room. We therefore combine "room words" with "problem words", plus a list of
# specific issues that count on their own (e.g. air conditioning, noise).
# Stems (e.g. "δωμάτι", "κλιματισμ") are used so accented forms also match.
ROOM_WORDS = [
    "room", "δωμάτι",
]

ROOM_PROBLEM_WORDS = [
    "problem", "issue", "broken", "not working", "doesn't work", "dirty",
    "not clean", "messy",
    "πρόβλημα", "βλάβη", "δεν δουλεύει", "δεν λειτουργεί", "δεν είναι καθαρό",
    "βρόμικο", "ακάθαρτο", "καθαριότητα",
]

# Specific issues that signal a room problem on their own.
SPECIFIC_ROOM_ISSUE_KEYWORDS = [
    "air conditioning", "a/c", "heating", "noise", "noisy",
    "κλιματισμ", "θόρυβο", "θορυβ",
]


def _is_room_issue(text: str) -> bool:
    """Return True if the (lowercased) text describes a problem with a room."""
    if _contains_any(text, SPECIFIC_ROOM_ISSUE_KEYWORDS):
        return True
    # A room mention plus a problem word, or a room mention plus an urgent/safety
    # problem (leak, no power, etc.) both count as a room issue.
    has_room = _contains_any(text, ROOM_WORDS)
    has_problem = _contains_any(text, ROOM_PROBLEM_WORDS) or _contains_any(
        text, URGENT_SAFETY_KEYWORDS
    )
    return has_room and has_problem

TRANSPORT_KEYWORDS = [
    "airport transfer", "airport", "taxi", "shuttle", "transfer", "port",
    "μεταφορά", "αεροδρόμιο", "ταξί", "λιμάνι",
]

# Things we must never handle directly (private/sensitive information).
SENSITIVE_KEYWORDS = [
    "credit card", "card number", "password", "another guest",
    "someone else", "private data", "personal data", "passport number",
    "κάρτα", "κωδικός", "προσωπικά δεδομένα", "άλλου πελάτη", "διαβατήριο",
]

# Broad/unclear phrases that need a clarification question.
AMBIGUOUS_PHRASES = [
    "i need help", "help me", "i have a problem", "i want to cancel",
    "can you help", "i have a question",
    "χρειάζομαι βοήθεια", "έχω πρόβλημα", "θέλω να ακυρώσω", "βοήθεια",
]


def _contains_any(text: str, keywords: list) -> bool:
    """Return True if any keyword is found in the (already lowercased) text."""
    return any(keyword in text for keyword in keywords)


def classify_intent(message: str) -> str:
    """Classify the message into one hotel guest support intent using keywords."""
    text = message.lower().strip()

    # Sensitive/private requests take top priority for safety.
    if _contains_any(text, SENSITIVE_KEYWORDS):
        return "sensitive_private"

    # Room issues are checked before the generic "ambiguous" phrases, so a
    # message like "Έχω πρόβλημα στο δωμάτιο" is treated as a room issue rather
    # than a vague "I have a problem".
    if _is_room_issue(text):
        return "room_issue"

    # Broad/unclear messages should be treated as ambiguous.
    if _contains_any(text, AMBIGUOUS_PHRASES):
        return "ambiguous"

    # Specific hotel guest support topics.
    if _contains_any(text, CANCELLATION_REFUND_KEYWORDS):
        return "cancellation_refund"
    if _contains_any(text, CHECKIN_CHECKOUT_KEYWORDS):
        return "check_in_checkout"
    if _contains_any(text, BOOKING_KEYWORDS):
        return "booking"
    if _contains_any(text, AMENITIES_KEYWORDS):
        return "amenities"
    if _contains_any(text, TRANSPORT_KEYWORDS):
        return "transport"
    if _contains_any(text, PAYMENTS_KEYWORDS):
        return "payments"

    # Nothing matched -> not a hotel guest support topic we recognize.
    return "out_of_scope"


# --- Ambiguity detection ----------------------------------------------------


def is_ambiguous(message: str, intent: str) -> bool:
    """Return True for messages that need a clarification question.

    Room issues are routed to a clarification too: the knowledge base does not
    contain per-room fixes, so the helpful next step is to ask the guest what is
    wrong (air conditioning, noise, cleaning, etc.) before escalating to staff.
    """
    if intent in ("ambiguous", "room_issue"):
        return True
    text = message.lower().strip()
    return _contains_any(text, AMBIGUOUS_PHRASES)


# --- Escalation detection ---------------------------------------------------

# Words that suggest an angry/upset guest or a serious issue.
ESCALATION_KEYWORDS = [
    "double charge", "charged twice", "fraud", "scam", "legal", "lawyer",
    "complaint", "angry", "furious", "unacceptable", "terrible", "worst",
    "speak to a manager", "human",
    "διπλή χρέωση", "απάτη", "νομικ", "καταγγελία", "θυμωμέν", "απαράδεκτο",
    "διευθυντή", "άνθρωπο",
]

# Urgent / safety wording for room issues that should go straight to staff
# instead of a clarification (no electricity, water leak, fire, etc.).
URGENT_SAFETY_KEYWORDS = [
    "water leak", "leaking", "leak", "flood", "flooding",
    "no electricity", "no power", "power outage", "fire", "smoke", "gas leak",
    "τρέχει νερό", "διαρροή", "πλημμύρα", "δεν έχουμε ρεύμα", "δεν έχει ρεύμα",
    "χωρίς ρεύμα", "φωτιά", "καπνός", "διαρροή αερίου",
]


def needs_escalation(message: str, intent: str) -> bool:
    """Return True when the issue should be handed off to hotel staff."""
    # Sensitive/private requests always require a human.
    if intent == "sensitive_private":
        return True

    text = message.lower().strip()

    # Angry/upset wording or serious issues (fraud, legal, complaint, etc.).
    if _contains_any(text, ESCALATION_KEYWORDS):
        return True

    # Urgent or safety-related room issues (leaks, no power, fire, etc.).
    if _contains_any(text, URGENT_SAFETY_KEYWORDS):
        return True

    return False


# --- Main agent flow --------------------------------------------------------


def _resolve_language(requested_language: str, message: str) -> str:
    """Respect an explicit "en"/"el" choice, otherwise auto-detect."""
    if requested_language in ("en", "el"):
        return requested_language
    return detect_language(message)


def _build_response(
    answer,
    decision,
    language,
    intent,
    sources=None,
    evidence_level="not_checked",
    escalation_summary=None,
):
    """Helper to build the structured response dictionary."""
    return {
        "answer": answer,
        "decision": decision,
        "language": language,
        "intent": intent,
        "sources": sources if sources is not None else [],
        "evidence_level": evidence_level,
        "escalation_summary": escalation_summary,
    }


# --- Grounded answer generation ---------------------------------------------


# Shown only when the Gemini call itself fails (e.g. network/quota error). It is
# deliberately different from the knowledge-base fallback: here we DID find
# relevant context, so we must not claim the information is unavailable.
_GENERATION_ERROR_MESSAGE = {
    "en": (
        "Sorry, I'm having trouble generating a reply right now. Please try again "
        "in a moment, or contact our hotel staff for help."
    ),
    "el": (
        "Συγγνώμη, αντιμετωπίζω πρόβλημα στη δημιουργία απάντησης αυτή τη στιγμή. "
        "Παρακαλώ δοκιμάστε ξανά σε λίγο ή επικοινωνήστε με το προσωπικό του "
        "ξενοδοχείου."
    ),
}


def generate_grounded_answer(message: str, language: str, retrieved_chunks: list) -> str:
    """Use Gemini to answer the question using the retrieved hotel context.

    The context here has already passed the retrieval threshold, so it is
    relevant. The prompt tells Gemini to actually use it and NOT to claim the
    information is unavailable when it is present.

    If the Gemini call fails (network/quota), we return a short "try again"
    message rather than a "no information" message, because we did find context.
    """
    # Build a clearly labelled context block from the retrieved chunks.
    context = "\n\n".join(
        f"[source: {chunk['source']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )

    language_name = "Greek" if language == "el" else "English"
    user_prompt = (
        "You are given hotel support context below. Use the context to answer the "
        "guest's question.\n"
        "Rules:\n"
        "- Do NOT say that information is unavailable if the context contains "
        "relevant information.\n"
        "- If the context includes only partial information, answer with what is "
        "available and mention that the guest can contact hotel staff for "
        "booking-specific details.\n"
        "- Only say \"I don't have that information\" if the context truly does not "
        "contain anything relevant.\n"
        "- Do NOT invent AegeanStay Hotels policies beyond the context.\n"
        f"- Answer in {language_name}.\n"
        "- Keep the answer concise and customer-support friendly.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{message}"
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=user_prompt,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        answer = (response.text or "").strip()
        # If Gemini returns nothing usable, show the transient-error message.
        if not answer:
            return _GENERATION_ERROR_MESSAGE[language]
        return answer
    except Exception:
        # Network/quota error -> honest "try again" message, app stays up.
        return _GENERATION_ERROR_MESSAGE[language]


# --- Guardrail: prevent fallback-style answers when evidence is strong -------
# Sometimes the model returns a "I don't have that information" style answer (or
# generation fails) even though RAG retrieved clearly relevant context. When the
# evidence is strong, that is wrong. These helpers detect such answers and build
# a simple answer straight from the retrieved context instead, so the demo stays
# stable and reliable.

# Phrases that signal an unwanted fallback-style answer (checked lowercased).
_BAD_FALLBACK_PHRASES = [
    "i don't have that information",
    "i do not have that information",
    "don't have that information",
    "don't have this information",
    "do not have this information",
    "information is not available",
    "i'm having trouble generating",  # our transient-generation message
    "δεν έχω αυτή την πληροφορία",
    "δεν βρήκα αυτή την πληροφορία",
    "δεν είναι διαθέσιμη",
    "αντιμετωπίζω πρόβλημα στη δημιουργία",  # our transient-generation message
]

_EXTRACTIVE_LEAD_IN = {
    "en": "Based on the hotel support information:",
    "el": "Με βάση τις πληροφορίες υποστήριξης του ξενοδοχείου:",
}

# Small stopword list so keyword matching focuses on meaningful words.
_STOPWORDS = {
    "the", "and", "for", "you", "your", "with", "that", "this", "are", "can",
    "does", "what", "when", "how", "from", "have", "not", "our", "but", "please",
    "και", "για", "την", "τον", "της", "στο", "στη", "με", "να", "το", "τι",
    "είναι", "από", "σας", "μου", "ένα", "μια",
}


def is_bad_fallback_answer(answer: str) -> bool:
    """Return True if the answer looks like an unwanted fallback message."""
    if not answer:
        return True
    text = answer.lower()
    return any(phrase in text for phrase in _BAD_FALLBACK_PHRASES)


def _tokens(text: str) -> list:
    """Lowercased words of length >= 3 that are not stopwords (EN/EL aware)."""
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


def _relevance(question_tokens: set, sentence: str) -> int:
    """Count how many question keywords appear in the sentence.

    Uses simple prefix matching so word variants count (e.g. the question word
    "cancel" matches "cancelled" / "cancellation" in the text).
    """
    sentence_tokens = _tokens(sentence)
    score = 0
    for q in question_tokens:
        for s in sentence_tokens:
            if q == s or (len(q) >= 4 and (s.startswith(q) or q.startswith(s))):
                score += 1
                break
    return score


def _split_sentences(text: str) -> list:
    """Split text into clean sentences, dropping markdown headings/labels."""
    raw = re.split(r"(?<=[.!?;])\s+|\n+", text)
    sentences = []
    for s in raw:
        s = s.strip().strip("#*-—").strip()
        # Skip empties, markdown headings, title lines, and short labels.
        if not s or s.startswith("#") or "—" in s or len(s) < 15:
            continue
        sentences.append(s)
    return sentences


def build_extractive_answer(message: str, language: str, retrieved_chunks: list) -> str:
    """Build a simple answer directly from the most relevant retrieved context.

    Used as a guardrail when Gemini returns a fallback-style answer despite
    strong retrieval evidence. It does not invent anything: it just selects the
    most relevant sentences (in the guest's language) from the retrieved chunks.
    The question keywords drive the selection, which naturally covers the common
    demo questions (check-in, breakfast, airport transfer, cancellation).
    """
    question_tokens = set(_tokens(message))

    # Collect candidate sentences in the guest's language, keeping their order.
    candidates = []  # (score, order_index, sentence)
    order = 0
    for chunk in retrieved_chunks:
        for sentence in _split_sentences(chunk["text"]):
            if detect_language(sentence) != language:
                continue
            score = _relevance(question_tokens, sentence)
            candidates.append((score, order, sentence))
            order += 1

    # Prefer sentences that share keywords with the question (best first).
    matched = sorted(
        (c for c in candidates if c[0] > 0), key=lambda c: (-c[0], c[1])
    )[:4]

    # If nothing matched, fall back to the first sentences of the top context.
    chosen = matched if matched else candidates[:2]

    # Present in original order and de-duplicate.
    chosen = sorted(chosen, key=lambda c: c[1])
    seen = set()
    sentences = []
    for _, _, sentence in chosen:
        if sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)
    sentences = sentences[:4]

    lead = _EXTRACTIVE_LEAD_IN.get(language, _EXTRACTIVE_LEAD_IN["en"])
    body = " ".join(sentences).strip()
    if not body:
        # Absolute last resort: a trimmed slice of the top chunk.
        body = retrieved_chunks[0]["text"][:300].strip()
    return f"{lead} {body}"


def run_agent(message: str, requested_language: str = "auto") -> dict:
    """Run the rule-based agent flow and return a structured decision."""
    language = _resolve_language(requested_language, message)
    intent = classify_intent(message)

    # 1) Sensitive/private requests -> escalate with the sensitive template.
    if intent == "sensitive_private":
        return _build_response(
            answer=get_template("sensitive", language),
            decision="escalate",
            language=language,
            intent=intent,
            escalation_summary=(
                f"Guest issue may require hotel staff review. Intent: {intent}."
            ),
        )

    # 2) Other escalation triggers (angry, fraud, legal, complaint, etc.).
    if needs_escalation(message, intent):
        return _build_response(
            answer=get_template("escalation", language),
            decision="escalate",
            language=language,
            intent=intent,
            escalation_summary=(
                f"Guest issue may require hotel staff review. Intent: {intent}."
            ),
        )

    # 3) Ambiguous messages -> ask a clarification question.
    # Room issues get a room-specific clarification; everything else the generic one.
    if is_ambiguous(message, intent):
        template_key = "room_clarification" if intent == "room_issue" else "clarification"
        return _build_response(
            answer=get_template(template_key, language),
            decision="clarify",
            language=language,
            intent=intent,
        )

    # 4) Out-of-scope messages -> graceful fallback.
    if intent == "out_of_scope":
        return _build_response(
            answer=get_template("fallback", language),
            decision="fallback",
            language=language,
            intent=intent,
        )

    # 5) Recognized support topic -> retrieve context and answer with Gemini.
    # If retrieval fails (e.g. missing key or network), treat it as no results.
    try:
        retrieved_chunks = retrieve(message, top_k=3)
    except Exception:
        retrieved_chunks = []

    top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
    retrieved_sources = {chunk["source"] for chunk in retrieved_chunks}

    # Cancellation/refund questions are important to answer. If the retrieval is
    # only slightly below the normal threshold but it did pull in the right doc
    # (cancellations_refunds.md), we still allow a grounded answer.
    cancellation_override = (
        intent == "cancellation_refund"
        and "cancellations_refunds.md" in retrieved_sources
        and top_score >= 0.30
    )

    # Not enough relevant context -> graceful fallback instead of guessing.
    if not retrieved_chunks or (top_score < 0.45 and not cancellation_override):
        return _build_response(
            answer=get_template("fallback", language),
            decision="fallback",
            language=language,
            intent=intent,
            sources=[],
            evidence_level="low",
        )

    # We have good context -> generate a grounded answer.
    answer = generate_grounded_answer(message, language, retrieved_chunks)

    # How confident are we, based on the best matching chunk?
    evidence_level = "high" if top_score >= 0.60 else "medium"

    # Collect the unique source filenames we used (keep their order).
    sources = []
    for chunk in retrieved_chunks:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    # Guardrail: the retrieval evidence is strong (high/medium) and we have
    # sources, so a fallback-style answer here is wrong. If Gemini returned one
    # (or generation failed), answer directly from the retrieved context.
    if evidence_level in ("high", "medium") and sources and is_bad_fallback_answer(answer):
        answer = build_extractive_answer(message, language, retrieved_chunks)

    return _build_response(
        answer=answer,
        decision="answer",
        language=language,
        intent=intent,
        sources=sources,
        evidence_level=evidence_level,
    )
