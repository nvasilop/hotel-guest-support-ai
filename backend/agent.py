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
    "cancel", "cancellation", "refund", "no-show", "no show",
    "ακύρωση", "ακυρώσω", "επιστροφή χρημάτων", "επιστροφή χρημ",
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


def generate_grounded_answer(message: str, language: str, retrieved_chunks: list) -> str:
    """Use Gemini to answer the question using ONLY the retrieved chunks.

    If Gemini fails for any reason, we return a polite fallback message so the
    app keeps running instead of crashing.
    """
    # Build a single context string from the retrieved knowledge base chunks.
    context = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )

    # Tell Gemini exactly how to behave: stay grounded, no invented policies,
    # answer in the guest's language, and be concise and friendly.
    language_name = "Greek" if language == "el" else "English"
    user_prompt = (
        "Use ONLY the AegeanStay Hotels guest support context below to answer the "
        "guest's question.\n"
        "- Do NOT invent AegeanStay Hotels policies or details that are not in the "
        "context.\n"
        "- If the context does not contain the answer, say the information is not "
        "available and recommend contacting the hotel staff.\n"
        f"- Answer in {language_name}.\n"
        "- Keep the answer concise, polite, and guest-support friendly.\n\n"
        f"Context:\n{context}\n\n"
        f"Guest question: {message}"
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=user_prompt,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        answer = (response.text or "").strip()
        # If Gemini returns nothing usable, fall back gracefully.
        if not answer:
            return get_template("fallback", language)
        return answer
    except Exception:
        # Any API/network error -> polite fallback, app stays up.
        return get_template("fallback", language)


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

    # Not enough relevant context -> graceful fallback instead of guessing.
    if not retrieved_chunks or top_score < 0.45:
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

    return _build_response(
        answer=answer,
        decision="answer",
        language=language,
        intent=intent,
        sources=sources,
        evidence_level=evidence_level,
    )
