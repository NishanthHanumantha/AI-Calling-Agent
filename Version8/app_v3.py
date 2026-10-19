import os
import re
import math
import html
import traceback
import fitz
import requests
from dotenv import load_dotenv

load_dotenv()

from collections import Counter
from fastapi import FastAPI, Form, Request
from fastapi.responses import Response

app = FastAPI()

# =====================================================
# CONFIG
# =====================================================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
MODEL = "sarvam-105b-conversations"
BROCHURE_FILE = "sobha-townpark-brochure.pdf"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BROCHURE_PATH = os.path.join(BASE_DIR, BROCHURE_FILE)
MAX_FAQ_TURNS = 8
LOW_RETRIEVAL_SCORE = 0.12

# Polly.Aditi is the most widely supported Indian-English voice on Twilio.
# Set TWILIO_SAY_VOICE=Polly.Aditi-Neural in the environment only if your account supports it.
TWILIO_SAY_VOICE = os.getenv("TWILIO_SAY_VOICE", "Polly.Aditi")

# Do not set language on <Say> for Polly — some accounts error with language+voice combo ("application error").

# Gather: longer wait + fixed pause after speech reduces early cut-off from background noise
GATHER_TIMEOUT = "30"
GATHER_SPEECH_TIMEOUT = "5"

MAX_EMPTY_RETRIES = 3

# Canonical visit slots (spoken + STT hints)
VISIT_SLOTS_SPOKEN = "ten A M, eleven thirty A M, four P M, or six P M"
BOOKABLE_SLOTS = ("10:00 AM", "11:30 AM", "4:00 PM", "6:00 PM")

# Twilio speech hints improve STT accuracy for expected customer words
SPEECH_HINTS = (
    "amenities, pricing, location, floor plan, configuration, "
    "site visit, yes, no, ten am, eleven thirty am, four pm, six pm"
)

# =====================================================
# PROJECT KNOWLEDGE BASE (used when brochure lacks details)
# =====================================================

PROJECT_KB = {
    "project_name": "SOBHA Townpark",
    "brand": "Sobha Limited",
    "city": "Bengaluru",
    "theme": "New York-themed luxury apartments",
    "location": "Near Electronic City on Hosur Road, Bengaluru",
    "starting_price": "INR 1.8 Crore onwards",
    "configurations": "1 BHK, 2 BHK, 3 BHK and 4 BHK apartments",
    "amenities": (
        "clubhouses, swimming pools, sports courts, landscaped gardens, "
        "kids play areas, forest grove and camping grounds"
    ),
    "possession": "Please check with our sales team for the latest possession timeline",
    "payment": "Flexible payment plans and home loan assistance are available",
}

# Brochure super-built-up sizes, one row per configuration, so a 3 BHK reply cannot borrow a 2 BHK size.
UNIT_SIZES = {
    "1 BHK": "1 BHK Luxe is 754 sq. ft.",
    "2 BHK": "2 BHK Luxe is 1240 to 1248 sq. ft. 2 BHK Grande is 1337 to 1678 sq. ft.",
    "3 BHK": "3 BHK Luxe is 1506 to 1514 sq. ft. 3 BHK Grande is 1842 to 2547 sq. ft.",
    "4 BHK": "4 BHK Grande is 2203 to 2846 sq. ft.",
}

# =====================================================
# MEMORY STORE
# =====================================================

conversation_memory = {
    "stage": "greeting",
    "faq_count": 0,
    "last_intent": None,
    "visit_preference": None,
    "awaiting_visit_decision": False,
    "visit_day": None,
    "selected_slot": None,
    "empty_retry_count": 0,
    "slot_pick_retries": 0,
    "customer_locality": None,
    "customer_configuration": None,
    "last_agent_question": None,
    "configuration_asked": False,
    "customer_variant": None,
    "size_decided": False,
    "size_closed_before_turn": False,
    "last_spoken_reply": None,
    "declined_topics": [],
    "covered_topics": [],
    "open_topics": [],
    "customer_purpose": None,
    "visit_offered": False,
    "fallback_asked": False,
    "theme_spoken": False,
    "price_stated": False,
    "payment_stated": False,
    "open_project_asked": False,
    "awaiting_callback": False,
}

# =====================================================
# STOPWORDS
# =====================================================

STOPWORDS = {
    "the", "and", "is", "in", "at", "of", "for", "to", "with",
    "a", "an", "our", "from", "on", "that", "it", "are", "this",
    "as", "be", "by", "or", "was", "will", "have", "has", "what",
    "how", "tell", "me", "about", "you", "your", "can", "could",
    "would", "like", "know", "please", "want", "any", "more",
}

INTENT_KEYWORDS = {
    "amenities": [
        "amenit", "clubhouse", "club house", "pool", "swimming",
        "kids", "gym", "sports", "wellness", "garden", "landscape",
        "play area", "jogging", "tennis", "badminton", "spa",
        "fitness", "recreation", "facilities",
    ],
    "pricing": [
        "price", "cost", "rate", "budget", "pricing", "offer",
        "payment", "emi", "sqft", "square feet", "per sq", " lakh",
        " crore", "affordable", "expensive", "how much",
    ],
    "location": [
        "location", "where", "hosur", "electronic city", "connectivity",
        "near", "nearby", "road", "airport", "school", "hospital",
        "office", "it hub", "metro", "distance", "address", "bengaluru",
        "bangalore",
    ],
    "floorplan": [
        "floor", "plan", "layout", "configuration", "configurations",
        "bhk", "bedroom", "sq ft", "sqft", "carpet", "super built",
        "unit", "apartment size", "1 bhk", "2 bhk", "3 bhk", "4 bhk",
    ],
}

INTENT_QUERY_EXPANSION = {
    "amenities": "clubhouse swimming pool sports gym kids play area wellness amenities facilities",
    "pricing": "price cost rate pricing payment offer per square feet starting price",
    "location": "location Electronic City Hosur Road Bengaluru connectivity nearby schools offices",
    "floorplan": "floor plan layout 1 BHK 2 BHK 3 BHK configuration carpet area",
    "general": "SOBHA Townpark luxury residential apartment New York themed",
}

CANONICAL_NO_INFO = (
    "I'm sorry, I don't have that detail right now; our sales team can share the latest information."
)

# Common Twilio STT mis-hearings after the qualify prompt
STT_MISHEAR_MAP = {
    r"^(yes[, ]*)?(it is|it's|its|it\'s)\.?$": "amenities",
    r"^(yes[, ]*)?(a menities|amenties|amenity)\.?$": "amenities",
    r"^pricing\.?$": "pricing",
    r"^location\.?$": "location",
    r"^(floor plan|floorplan|configuration|configurations)\.?$": "floor plan",
}

# =====================================================
# PDF CLEANING & CHUNKING
# =====================================================


def clean_pdf_text(text):
    """Strip noisy page headers, footers, and normalize whitespace."""
    text = re.sub(r"\s+", " ", text)
    junk_patterns = [
        r"CONTENTS",
        r"SOBHA Legacy",
        r"Neighbourhood",
        r"Master Plan",
        r"Interiors",
        r"Testimonials",
        r"Artist.?s Impression",
        r"Concept Image",
        r"PORTFOLIO OF EXCELLENCE",
        r"PASSION\. INKED WITH PERFECTION",
        r"\d+ of \d+",
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def load_pdf_text(pdf_path):
    """Load all brochure text from the PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text("text")
        pages.append(clean_pdf_text(page_text))
    doc.close()
    return "\n".join(pages)


def chunk_text(text, chunk_size=900, overlap=150):
    """Create overlapping chunks that preserve context for retrieval."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_words = current_chunk.strip().split()[-max(1, overlap // 6):]
            current_chunk = f"{' '.join(overlap_words)} {sentence}".strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if len(chunk) > 60]


def tokenize(text):
    """Normalize text into tokens for similarity scoring."""
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9']+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def build_tfidf_vectors(chunks):
    """Build normalized TF-IDF vectors for brochure chunks."""
    doc_tokens = [tokenize(chunk) for chunk in chunks]
    doc_count = len(doc_tokens)
    df = Counter()

    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1

    idf = {
        term: math.log((1 + doc_count) / (1 + count)) + 1.0
        for term, count in df.items()
    }

    vectors = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        vec = {term: freq * idf.get(term, 0.0) for term, freq in tf.items()}
        length = math.sqrt(sum(value * value for value in vec.values()))
        vectors.append(
            {term: value / length for term, value in vec.items()} if length > 0 else {}
        )

    return vectors


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two sparse vectors."""
    if not vec1 or not vec2:
        return 0.0
    return sum(vec1.get(term, 0.0) * vec2.get(term, 0.0) for term in vec1)


def keyword_overlap_score(query_tokens, chunk_text):
    """Score chunk by direct keyword overlap with the user query."""
    chunk_tokens = set(tokenize(chunk_text))
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = sum(1 for token in query_tokens if token in chunk_tokens)
    return overlap / len(query_tokens)


def intent_keyword_score(intent, chunk_text):
    """Boost chunks that contain intent-specific brochure terms."""
    chunk_lower = chunk_text.lower()
    keywords = INTENT_KEYWORDS.get(intent, [])
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in chunk_lower)
    return hits / len(keywords)


# =====================================================
# BROCHURE PREPARATION (never crash startup if PDF missing)
# =====================================================

try:
    pdf_text = load_pdf_text(BROCHURE_PATH)
    brochure_chunks = chunk_text(pdf_text)
    brochure_vectors = build_tfidf_vectors(brochure_chunks)
except Exception as brochure_exc:
    print("BROCHURE LOAD FAILED:", brochure_exc)
    traceback.print_exc()
    pdf_text = ""
    brochure_chunks = []
    brochure_vectors = []

print("=" * 50)
print("BROCHURE LOADED")
print("TOTAL CHUNKS:", len(brochure_chunks))
print("=" * 50)

# =====================================================
# STT CORRECTION
# =====================================================


def correct_stt(user_text, stage=None):
    """
    Fix common Twilio speech-to-text errors using stage context.
    Example: customer says 'AMENITIES' but STT returns 'It is.'
    """
    original = user_text.strip()
    if not original:
        return original

    normalized = re.sub(r"\s+", " ", original.lower()).strip()
    if re.search(r"\bgranded\b", normalized):
        original = re.sub(r"(?i)\bgranded\b", "Grande", original)
        normalized = normalized.replace("granded", "grande")

    last_question = (conversation_memory.get("last_agent_question") or "").lower()
    if is_bare_configuration_word(original):
        return original
    if re.match(r"^(configuration|configurations)\.?$", normalized) and re.search(
        r"\b(configuration|luxe|grande|layout)\b", last_question
    ):
        return original

    for pattern, replacement in STT_MISHEAR_MAP.items():
        if re.match(pattern, normalized, flags=re.IGNORECASE):
            print(f"STT CORRECTION: '{original}' -> '{replacement}'")
            return replacement

    # After the qualify prompt, a short topic label is usually a menu pick.
    # Do not rewrite an actual question or concern into that label.
    if stage in ("qualify", "answer_faq") and len(normalized.split()) <= 4:
        if extract_configuration(original):
            return original
        if re.search(r"\b(luxe|grande)\b", normalized):
            return original
        if re.search(r"\b(size|range|luxe|grande|sq\. ft|square feet)\b", last_question) and re.search(
            r"\d{3,4}|square|sq\.?\s*ft|luxe|grande", normalized
        ):
            return original
        if re.search(r"\b(what|how|why|when|where|who|offer|expensive|far|distance)\b|\?", normalized):
            return original
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                topic = {
                    "amenities": "amenities",
                    "pricing": "pricing",
                    "location": "location",
                    "floorplan": "floor plan and configurations",
                }.get(intent)
                if topic:
                    print(f"STT TOPIC DETECTED: '{original}' -> intent '{intent}'")
                    return f"tell me about {topic}"

    return original


def say_block(text, escape_content=True):
    """TTS with Polly Indian English (no language attr — avoids Twilio/Polly errors)."""
    inner = escape_twiml(text) if escape_content else text
    return f'<Say voice="{TWILIO_SAY_VOICE}">{inner}</Say>'


def gather_block(prompt_text):
    """Gather with hints, en-IN, and timeouts that tolerate light background noise."""
    remember_spoken(prompt_text)
    safe = escape_twiml(prompt_text)
    hints_attr = html.escape(SPEECH_HINTS, quote=True)
    return f"""
<Gather
input="speech"
action="https://s-ai-calling-agent.duckdns.org/handle-speech"
method="POST"
actionOnEmptyResult="true"
timeout="{GATHER_TIMEOUT}"
speechTimeout="5" speechModel="experimental_utterances"
hints="{hints_attr}"
language="en-IN">
{say_block(safe, escape_content=False)}
</Gather>
"""


def visit_schedule_question():
    """Single wording for 'would you like to schedule a visit' (avoid asking twice)."""
    return "Would you like to schedule a site visit to SOBHA Townpark?"


def parse_bookable_slot(speech):
    """Map customer speech to one of the four fixed slots."""
    if not speech or not speech.strip():
        return None
    t = speech.lower().strip()
    t = re.sub(r"[^\w\s:./]", " ", t)
    t = re.sub(r"\s+", " ", t)

    if re.search(r"11\s*[:.]?\s*30|eleven\s*thirty|11\s*30", t):
        return "11:30 AM"
    if "ten" in t or re.search(r"\b10\b", t):
        return "10:00 AM"
    if "six" in t or re.search(r"\b6\s*(pm)?\b", t) or re.search(r"\b18\s*h", t):
        return "6:00 PM"
    if "four" in t or re.search(r"\b4\s*(pm)?\b", t) or re.search(r"\b16\s*h", t):
        return "4:00 PM"

    if "first" in t or "earliest" in t:
        return "10:00 AM"
    if "second" in t:
        return "11:30 AM"
    if "third" in t:
        return "4:00 PM"
    if "last" in t or "fourth" in t or "final" in t:
        return "6:00 PM"

    return None


def names_a_day(text):
    """True when the customer named a visit day, not some other sentence."""
    return chosen_day(text) is not None


def chosen_day(text):
    """The day they named, without the rest of the sentence."""
    match = re.search(
        r"\b((?:this|next)\s+)?(today|tomorrow|weekend|weekends|weekday|weekdays|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        (text or "").lower(),
    )
    if not match:
        return None
    prefix = (match.group(1) or "").strip()
    day = match.group(2)
    if day == "weekends":
        day = "weekend"
    elif day == "weekdays":
        day = "weekday"
    elif day not in {"today", "tomorrow", "weekend", "weekday"}:
        day = day.capitalize()
    if prefix:
        return f"{prefix} {day}"
    return day


# =====================================================
# PROJECT KB HELPERS
# =====================================================


def kb_context_for_intent(intent):
    """Build structured knowledge-base context for the LLM."""
    lines = [
        f"Project: {PROJECT_KB['project_name']}",
        f"Brand: {PROJECT_KB['brand']}",
        f"Theme: {PROJECT_KB['theme']}",
        f"Location: {PROJECT_KB['location']}",
        f"Starting price: {PROJECT_KB['starting_price']}",
        f"Configurations: {PROJECT_KB['configurations']}",
        f"Amenities: {PROJECT_KB['amenities']}",
        f"Payment: {PROJECT_KB['payment']}",
    ]
    if intent == "pricing":
        lines.insert(0, "Use starting price and payment details for pricing questions.")
    elif intent == "amenities":
        lines.insert(0, "Use amenities list for amenities questions.")
    elif intent == "location":
        lines.insert(0, "Use location details for location questions.")
    elif intent == "floorplan":
        lines.insert(0, "Use configurations for layout and BHK questions.")
        stated = conversation_memory.get("customer_configuration")
        if stated in UNIT_SIZES:
            lines.insert(
                0,
                "Use only these sizes for the stated configuration. "
                f"{UNIT_SIZES[stated]}",
            )
        else:
            lines.insert(
                0,
                "Use only these sizes, and do not swap Luxe and Grande. "
                + " ".join(UNIT_SIZES[key] for key in ("1 BHK", "2 BHK", "3 BHK", "4 BHK")),
            )
    return "\n".join(lines)


def answer_from_kb(intent):
    """Direct spoken answer from project knowledge base (no LLM)."""
    if intent == "pricing":
        return (
            f"SOBHA Townpark starts at approximately {PROJECT_KB['starting_price']}, "
            f"depending on the configuration and floor. {PROJECT_KB['payment']}."
        )
    if intent == "amenities":
        return (
            f"SOBHA Townpark offers {PROJECT_KB['amenities']}, "
            "along with wellness and recreation spaces."
        )
    if intent == "location":
        return (
            f"SOBHA Townpark is located {PROJECT_KB['location']}, "
            "with excellent connectivity to Electronic City and major IT corridors."
        )
    if intent == "floorplan":
        return (
            f"The project offers {PROJECT_KB['configurations']}, "
            "with spacious layouts and premium finishes."
        )
    return (
        f"{PROJECT_KB['project_name']} is a {PROJECT_KB['theme']} project in "
        f"{PROJECT_KB['city']}, located {PROJECT_KB['location']}."
    )


def should_use_kb_primary(intent, top_score, brochure_context):
    """Use KB when brochure retrieval is weak or irrelevant."""
    if intent in ("pricing", "floorplan"):
        return top_score < LOW_RETRIEVAL_SCORE
    if top_score < LOW_RETRIEVAL_SCORE:
        return True
    if intent == "pricing" and not re.search(r"\d|lakh|crore|price|rs\.?", brochure_context, re.I):
        return True
    disclaimer_hits = len(re.findall(r"representation|warranty|general information", brochure_context, re.I))
    return disclaimer_hits >= 2 and top_score < 0.2


# =====================================================
# INTENT DETECTION & RETRIEVAL
# =====================================================


def detect_intent(user_query):
    """Detect FAQ topic intent from the user query."""
    query = user_query.lower()
    best_intent = "general"
    best_score = 0

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in query)
        if score > best_score:
            best_intent = intent
            best_score = score

    return best_intent


def build_retrieval_query(intent, user_query):
    """Combine user speech with intent hints for better brochure matching."""
    expansion = INTENT_QUERY_EXPANSION.get(intent, INTENT_QUERY_EXPANSION["general"])
    return f"{user_query} {expansion}".strip()


def retrieve_context(intent, user_query):
    """Retrieve brochure chunks; returns (context_text, top_score)."""
    retrieval_query = build_retrieval_query(intent, user_query)
    query_tokens = tokenize(retrieval_query)
    query_vector = build_tfidf_vectors([retrieval_query])[0]

    scored = []
    for index, chunk in enumerate(brochure_chunks):
        semantic_score = cosine_similarity(query_vector, brochure_vectors[index])
        overlap_score = keyword_overlap_score(query_tokens, chunk)
        intent_score = intent_keyword_score(intent, chunk)
        final_score = (semantic_score * 0.55) + (overlap_score * 0.25) + (intent_score * 0.20)
        scored.append((index, final_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    top_score = scored[0][1] if scored else 0.0

    top_chunks = []
    for index, score in scored:
        if score < 0.08:
            continue
        top_chunks.append(brochure_chunks[index])
        if len(top_chunks) >= 4:
            break

    if not top_chunks and scored:
        top_chunks = [brochure_chunks[scored[0][0]]]

    context = "\n\n".join(top_chunks).strip()
    print("=" * 50)
    print("RETRIEVAL INTENT:", intent)
    print("RETRIEVAL QUERY:", retrieval_query)
    print("TOP SCORE:", round(top_score, 4))
    print("CONTEXT PREVIEW:", context[:400])
    print("=" * 50)
    return context[:2200], top_score


THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"


def strip_reasoning_blocks(text):
    """Remove Sarvam chain-of-thought blocks and return any spoken answer."""
    if not text:
        return ""

    cleaned = text.strip()

    split_pattern = "(?i)" + re.escape(THINK_CLOSE) + "|" + re.escape("</think>")
    parts = re.split(split_pattern, cleaned)
    if len(parts) > 1:
        tail = parts[-1].strip()
        if tail:
            cleaned = tail

    thinking_patterns = [
        r"(?is)<think>.*?</think>",
        "(?is)" + re.escape(THINK_OPEN) + r".*?" + re.escape(THINK_CLOSE),
        r"(?is)<thinking>.*?</thinking>",
        r"(?is)<reasoning>.*?</reasoning>",
    ]
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, " ", cleaned)

    unclosed_patterns = [
        r"(?is)<think>.*$",
        "(?is)" + re.escape(THINK_OPEN) + r".*$",
        r"(?is)<thinking>.*$",
        r"(?is)<reasoning>.*$",
    ]
    for pattern in unclosed_patterns:
        cleaned = re.sub(pattern, " ", cleaned)

    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("\\n", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def looks_like_reasoning_only(text):
    """Detect responses that are still internal reasoning, not customer speech."""
    if not text:
        return True

    text_lower = text.lower()
    reasoning_starts = [
        "okay, let's",
        "okay lets",
        "let me ",
        "the user is asking",
        "the customer is asking",
        "the user asked",
        "first, looking at",
        "i need to go through",
        "i need to check",
        "looking at the",
        "let's tackle",
        "the brochure mentions",
        "the brochure doesn't",
        "the question is",
    ]
    if any(text_lower.startswith(prefix) for prefix in reasoning_starts):
        return True

    reasoning_markers = [
        "the user is asking",
        "the customer is asking",
        "brochure details",
        "provided brochure",
        "i need to go through",
        "let's tackle this",
    ]
    marker_hits = sum(1 for marker in reasoning_markers if marker in text_lower)
    return marker_hits >= 2


def clean_response(text):
    """Remove reasoning artifacts and keep the answer customer-ready."""
    if not text:
        return None

    text = strip_reasoning_blocks(text)

    if not text:
        return None

    if looks_like_reasoning_only(text):
        return None

    bad_phrases = [
        "according to the brochure",
        "based on the brochure",
        "from the context",
        "the context says",
        "as an ai",
        "as an assistant",
        "let me check",
        "let me look",
    ]
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in bad_phrases):
        return None

    return text.strip()


def fit_spoken_length(text, limit=320):
    """Keep whole sentences. A limit must not stop in the middle of a word or a question."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    kept = []
    total = 0
    for sentence in split_sentences(text):
        extra = len(sentence) + (1 if kept else 0)
        if total + extra > limit:
            break
        kept.append(sentence)
        total += extra
    if kept:
        return " ".join(kept)
    words = []
    for word in text.split():
        candidate = " ".join(words + [word])
        if len(candidate) > limit:
            break
        words.append(word)
    return " ".join(words).rstrip(",;:—-")


def fallback_response(intent):
    """Return answer from project knowledge base when brochure/LLM fails."""
    return answer_from_kb(intent)


def asks_about_offer(text):
    """True when the customer is asking for a commercial offer."""
    return re.search(r"\b(offer|discount|cashback)\b", (text or "").lower()) is not None


def utterance_needs_point_first(text):
    """Customer stated a constraint or concern, not a generic topic request."""
    return re.search(
        r"\b(too far|how far|distance|offer|discount|expensive|investment|own use|end use)\b",
        (text or "").lower(),
    ) is not None


def limit_to_one_question(text):
    """Keep at most the first spoken question."""
    if not text or text.count("?") <= 1:
        return text
    return text[: text.find("?") + 1].strip()


def contains_invented_commercial_claim(answer, supplied_context):
    """Reject discounts or offers that are not in the supplied project text."""
    banned = (
        "discount",
        "cashback",
        "free parking",
        "free registration",
        "launch offer",
        "limited-time offer",
        "special price",
        "appreciation",
    )
    low = (answer or "").lower()
    ctx = (supplied_context or "").lower()
    return any(term in low and term not in ctx for term in banned)


def point_first_fallback(user_query):
    """Grounded reply when a specific concern must not become a project dump."""
    text = (user_query or "").lower()
    if asks_about_offer(text):
        return (
            "I don't have a confirmed offer in the details I have. "
            "I can share the available project details if that would help."
        )
    if re.search(r"\b(too far|how far|distance)\b", text):
        return (
            "I understand, distance is the concern. "
            f"SOBHA Townpark is located {PROJECT_KB['location']}. "
            "I don't have a confirmed distance from the place you mentioned."
        )
    if "expensive" in text:
        return (
            "I understand, price is the concern. "
            f"SOBHA Townpark starts at approximately {PROJECT_KB['starting_price']}. "
            "I don't have a confirmed offer or discount in the details I have."
        )
    if "investment" in text:
        return (
            "I understand, you are looking at this as an investment. "
            f"SOBHA Townpark starts at approximately {PROJECT_KB['starting_price']}."
        )
    if re.search(r"\b(own use|end use)\b", text):
        return (
            "I understand, this is for your own use. "
            f"SOBHA Townpark is located {PROJECT_KB['location']}."
        )
    return (
        "I don't have that confirmed in the details I have. "
        "I can share the available project details if that would help."
    )


def safe_spoken_fallback(user_query, intent):
    """LLM failure path. Specific concerns stay specific. Other topics use the KB."""
    if asks_about_offer(user_query) or utterance_needs_point_first(user_query):
        print("USING POINT-FIRST FALLBACK")
        return point_first_fallback(user_query)
    print("USING PROJECT KB FALLBACK")
    return answer_from_kb(intent)


def generate_response(user_query, intent, brochure_context, top_score):
    """Answer the customer's point from supplied facts. Weak retrieval does not skip the model."""
    kb_section = kb_context_for_intent(intent)
    combined_context = f"Project knowledge:\n{kb_section}\n\nBrochure excerpts:\n{brochure_context}"
    if should_use_kb_primary(intent, top_score, brochure_context):
        print("WEAK BROCHURE MATCH — model must stay inside supplied facts")

    known = []
    if conversation_memory.get("customer_locality"):
        known.append(f"Locality already stated: {conversation_memory['customer_locality']}")
    if conversation_memory.get("customer_configuration"):
        known.append(f"Configuration already stated: {conversation_memory['customer_configuration']}")
    elif "configuration" in (conversation_memory.get("open_topics") or []):
        known.append("Configuration was left open. Do not ask which BHK again.")
    if conversation_memory.get("size_decided") and conversation_memory.get("customer_variant"):
        config = conversation_memory.get("customer_configuration") or "home"
        variant = conversation_memory["customer_variant"]
        size_line = UNIT_SIZES.get(config, "")
        known.append(
            f"Size already chosen: {variant} {config}. That choice is closed. {size_line} "
            "This is one range, not two separate sizes."
        )
    elif "size" in (conversation_memory.get("open_topics") or []):
        known.append("Size was left open. Do not ask about size, Luxe, Grande, or square footage again.")
    if "budget" in (conversation_memory.get("open_topics") or []):
        known.append("Budget was left open. Do not ask about budget or price.")
    known_line = "\n".join(known) if known else "No locality or configuration has been stated yet."
    last_question = conversation_memory.get("last_agent_question")
    if (
        conversation_memory.get("size_decided")
        and conversation_memory.get("size_closed_before_turn")
        and not customer_reopens_size(user_query)
    ):
        continuation = (
            f"You just asked: {last_question}\n"
            "The size is already decided. Do not ask about size, Luxe, Grande, layout, or square footage. "
            "Do not describe the home as two sizes. "
            "Acknowledge the customer and ask about a different point."
        )
    elif last_question and answers_size_question(user_query):
        continuation = (
            f"You just asked: {last_question}\n"
            "The customer is choosing one of those sizes. Confirm only that choice. "
            "Do not mention price or payment plans."
        )
    elif last_question and declining_last_offer(user_query):
        continuation = (
            f"You just asked: {last_question}\n"
            "The customer declined that. Acknowledge it and do not ask it again. "
            "Do not offer a site visit."
        )
    elif last_question and answering_last_question(user_query):
        continuation = (
            f"You just asked: {last_question}\n"
            "The customer is answering that question. "
            "Give the detail that question offered. Do not ask it again."
        )
    elif last_question:
        continuation = (
            f"You just asked: {last_question}\n"
            "Do not ask that question again, unless the customer asked you to wait. "
            "If they asked you to wait, say you will wait and repeat only that question."
        )
    else:
        continuation = "You have not asked a question yet."

    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are a conversational Indian real estate advisor on a phone call for SOBHA Townpark. "
        "Respond to the customer's actual point before any other project information. "
        "When it helps, begin with a brief acknowledgement of that point. "
        "Then answer only that point, using the project knowledge and brochure excerpts below. "
        "Add at most one relevant follow-up question, and only when a question is useful. "
        "A simple factual question can be only the answer. "
        "An objection needs acknowledgement, the grounded answer, and at most one question. "
        "If the customer needs time, acknowledge that and offer to share the available details. Do not push a site visit. "
        "If the customer asks you to wait, say you will wait and repeat only the question you just asked. "
        "Do not say goodbye and do not switch to the BHK menu. "
        "If they ask about an offer and the supplied information has no offer, say no confirmed offer is available. "
        "If a requested fact is not in the supplied information, say it is not confirmed. Do not guess. "
        "Do not invent discounts, launch offers, cashback, free parking, free registration, special prices, appreciation, or competitor comparisons. "
        "SOBHA Townpark is already located near Electronic City on Hosur Road. "
        "Do not ask which Bengaluru area they want unless they raise a commute or location concern. "
        "If they have already stated a configuration, do not ask for BHK again. "
        "If the customer is answering the question you just asked, continue that point. "
        "Do not ask that question again. A later question must be a different point. "
        "Use only the sizes listed for the configuration they already stated. "
        "If a size has already been chosen, do not ask about size again. "
        "Each Luxe or Grande line is one range, not two separate sizes. "
        "Do not mention price or payment plans unless the customer asked about price or payment. "
        "Do not give a generic amenities list unless they asked about amenities. "
        "Do not ask about open spaces, the clubhouse, or the swimming pool unless the customer asked about them. "
        "Speak a percent only when that percent is written in the project knowledge or the brochure excerpts. "
        "Do not say 'certainly' or 'I would be happy to'. "
        "Use short spoken sentences, not a written paragraph. "
        "At most three short sentences and at most one question mark. "
        "Reply with ONLY the words you will speak. No thinking tags, no bullet points. "
        "Do not mention brochures, documents, context, or AI. "
        "Do not invent facts that are not in the supplied information. "
        "You cannot book a site visit. Do not say a visit is booked or confirmed."
    )

    stage = conversation_memory.get("stage") or "greeting"
    user_prompt = (
        f"Customer said: {user_query}\n"
        f"Detected topic: {intent}\n"
        f"Current stage: {stage}\n"
        f"{known_line}\n"
        f"{continuation}\n\n"
        f"{combined_context}\n\n"
        "Speak only the reply to this customer."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 180,
        "top_p": 1,
        "reasoning_effort": None,
    }

    try:
        response = requests.post(
            SARVAM_URL,
            headers=headers,
            json=payload,
            timeout=20
        )
        response.raise_for_status()

        result = response.json()

        print("SARVAM RAW RESPONSE:")
        print(result)

        raw_answer = result["choices"][0]["message"]["content"]
        finish_reason = result["choices"][0].get("finish_reason", "stop")

        print("FINISH REASON:", finish_reason)
        print("RAW ANSWER PREVIEW:", raw_answer[:500])

        cleaned_answer = clean_response(raw_answer)

        if (
            not cleaned_answer
            or finish_reason == "length"
            or contains_invented_commercial_claim(cleaned_answer, combined_context)
        ):
            print("UNUSABLE MODEL REPLY -> SAFE FALLBACK")
            return safe_spoken_fallback(user_query, intent)

        spoken = leave_closed_size(user_query, drop_unasked_payment(user_query, cleaned_answer))
        spoken = drop_unsupported_percent(spoken, f"{kb_section}\n{brochure_context}")
        spoken = drop_unasked_amenity_question(user_query, spoken)
        spoken = leave_closed_configuration(user_query, spoken)
        spoken = drop_false_configuration_claim(spoken)
        spoken = leave_blocked_topics(spoken)
        spoken = drop_restated_location(spoken)
        spoken = drop_restated_theme(spoken)
        spoken = drop_question_just_answered(spoken)
        spoken = use_unit_sizes(user_query, spoken)
        spoken = restore_denied_sizes(spoken)
        spoken = drop_invented_floor_plan(spoken)
        spoken = limit_to_one_question(spoken) if spoken else ""
        spoken = fit_spoken_length(spoken) if spoken else ""
        return spoken or move_on_line()

    except Exception as exc:
        print("SARVAM ERROR:", str(exc))
        return safe_spoken_fallback(user_query, intent)

def contains_word(text, word):
    """Whole-word match so 'not' does not match inside 'configurations'."""
    return re.search(rf"\b{re.escape(word)}\b", text.lower()) is not None


def is_clear_negative(text):
    """Detect explicit refusal only (not substring matches)."""
    text_lower = text.lower().strip()
    if contains_word(text_lower, "no") and len(text_lower.split()) <= 3:
        return True
    negative_phrases = [
        "nope", "nah", "not now", "no thanks", "not interested",
        "no more questions", "nothing else", "that's all", "thats all",
    ]
    return any(phrase in text_lower for phrase in negative_phrases)


def is_not_interested(text):
    """The customer is declining the project, not asking to book a visit."""
    return re.search(r"\bnot interested\b", (text or "").lower()) is not None


def is_bare_okay(text):
    """A bare okay is not consent to book a visit."""
    cleaned = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
    return cleaned in {
        "ok", "okay", "ah okay", "ah ok", "okay okay", "ok ok", "ok okay", "okay ok",
    }


def approves_the_price(text):
    """A short approval of the price. This is not a request for another topic."""
    if is_bare_okay(text):
        return True
    cleaned = re.sub(r"[^\w\s]", "", (text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return re.fullmatch(
        r"(?:(?:okay|ok|yeah|yes)\s+)?(?:thats|that is|sounds|this is)\s+(?:good|great|fine|nice)",
        cleaned,
    ) is not None


def is_positive(text):
    """Detect whether the customer gave a positive response."""
    text_lower = text.lower()
    if is_unsure(text_lower) or is_flexible_about_size(text_lower):
        return False
    if is_clear_negative(text_lower):
        return False
    if re.search(r"\b(i would like|tell me|what about|how about|know about)\b", text_lower):
        return False
    return any(
        contains_word(text_lower, word)
        for word in ["yes", "yeah", "sure", "ok", "okay", "fine", "great", "yep", "yup"]
    )


def available_to_talk(text):
    """Greeting replies that mean the customer can continue, without the word yes."""
    text_lower = (text or "").lower()
    if is_clear_negative(text_lower) or asks_to_call_later(text_lower):
        return False
    phrases = (
        "can speak", "can talk", "speak now", "talk now", "good time",
        "go ahead", "i'm free", "im free", "i am free", "i'm available",
        "i am available", "listening", "carry on", "go on",
    )
    return any(phrase in text_lower for phrase in phrases)


def asks_to_call_later(text):
    """The customer cannot continue this call. A short hold is not a callback."""
    low = (text or "").lower()
    return re.search(
        r"\bbusy\b"
        r"|\bcall(?:\s+me)?\s+(?:later|back)\b"
        r"|\bcallback\b"
        r"|\bnot a good time\b"
        r"|\bin a meeting\b"
        r"|\bdriving\b"
        r"|\b(?:can(?:not|'t)|can not)\s+(?:talk|speak)\b",
        low,
    ) is not None


def wants_site_visit(text):
    """Detect whether the customer wants to schedule a visit."""
    text_lower = text.lower()
    visit_clues = [
        "site visit", "schedule", "book a visit", "appointment",
        "come and see", "visit the site", "book a slot",
    ]
    if any(clue in text_lower for clue in visit_clues):
        return True
    if contains_word(text_lower, "visit") and not re.search(r"\b(know|about|tell)\b", text_lower):
        return True
    return False


def asks_about_the_visit(text):
    """A question about the visit. A clear booking still uses wants_site_visit."""
    if wants_site_visit(text):
        return False
    low = (text or "").lower()
    if not re.search(r"\bvisit\b", low):
        return False
    if (
        is_clear_negative(text)
        or defers_the_visit(text)
        or declines_offered_topic(text)
        or declines_the_project(text)
    ):
        return False
    return True


def is_new_faq_question(text):
    """Detect that the customer is asking another project question."""
    text_lower = text.lower()
    question_clues = [
        "tell me", "i would like", "what about", "how about", "know about",
        "more about", "details on", "information on",
    ]
    if any(clue in text_lower for clue in question_clues):
        return True
    return detect_intent(text) != "general"


def is_done_with_questions(text):
    """Detect whether the customer has no more FAQ questions."""
    text_lower = text.lower()
    done_clues = [
        "no more questions", "nothing else", "that's all", "thats all",
        "i'm good", "im good", "all good", "enough questions",
    ]
    return any(clue in text_lower for clue in done_clues)


def needs_time(text):
    """Customer wants to decide later. Do not treat this as a visit request."""
    text_lower = (text or "").lower()
    clues = (
        "need to check",
        "i'll think",
        "i will think",
        "think about it",
        "discuss with my family",
        "check with my family",
        "talk to my family",
        "let me decide",
        "need some time",
    )
    return any(clue in text_lower for clue in clues)


ADVISOR_CLOSE = "I will have our sales advisor get in touch with you. Have a good day."
SHARE_CLOSE = "I will arrange to share the details of the project with you. Have a good day."


def refuses_details(text):
    """The customer does not want documents or any further project details."""
    return re.search(
        r"\b(don'?t need|do not need|no need|not need|don'?t want|do not want)"
        r".{0,24}\bdetails\b|\bno details\b",
        (text or "").lower(),
    ) is not None


def defers_the_visit(text):
    """The customer is putting the visit off, not asking for another topic."""
    return re.search(
        r"\b(let you know later|let me know later|i'?ll let you know|i will let you know|"
        r"(?:come|get) back(?: to you)? later)\b",
        (text or "").lower(),
    ) is not None


def has_not_finalized(text):
    """The customer has not settled the fact just asked."""
    return re.search(
        r"\b(haven'?t finalized|not finalized|yet to finalize)\b",
        (text or "").lower(),
    ) is not None


def is_good_for_now(text):
    """The customer is finished with the current topic, not necessarily the project."""
    return re.search(
        r"\b(i'm good|im good|i am good|good for now|all good)\b",
        (text or "").lower(),
    ) is not None


def finished_this_topic(text):
    return is_done_with_questions(text) or is_good_for_now(text)


def last_question_is_visit():
    """A yes books a day only when the question just asked was the visit."""
    last = (conversation_memory.get("last_agent_question") or "").lower()
    return re.search(r"\b(visit|schedule|which day|what time)\b", last) is not None


def last_question_offers_materials():
    """Highlights, documents, or details are not a visit question."""
    last = (conversation_memory.get("last_agent_question") or "").lower()
    if re.search(r"\b(visit|schedule|which day)\b", last):
        return False
    return re.search(
        r"\b(highlights|documents?|brochure|details|share)\b",
        last,
    ) is not None


def agrees_to_materials(text):
    return re.search(
        r"\b(share|send|documents?|brochure|highlights|details)\b",
        (text or "").lower(),
    ) is not None


def is_direct_project_question(text):
    """A specific project question should be answered, not replaced by qualification."""
    text_lower = (text or "").lower()
    if asks_about_offer(text_lower) or utterance_needs_point_first(text_lower):
        return True
    return re.search(
        r"\b(price|cost|how much|possession|amenities|configuration)\b",
        text_lower,
    ) is not None


def is_purpose_statement(text):
    """Investment or living in the home. A named area is a location, not this."""
    text_lower = (text or "").lower()
    if re.search(r"\b(investment|own use|end use|for myself|for my own)\b", text_lower):
        return True
    if not re.search(r"\b(to live|live in|for living)\b", text_lower):
        return False
    if extract_locality(text):
        return False
    last = (conversation_memory.get("last_agent_question") or "").lower()
    return "live in" in last or "investment" in last or "own use" in last


def is_bare_configuration_word(text):
    """BHK or configuration with no number is not a chosen home."""
    cleaned = re.sub(r"[^a-z\s]", " ", (text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned not in {"bhk", "bhks", "configuration", "configurations"}:
        return False
    if extract_configuration(text):
        return False
    if cleaned in {"configuration", "configurations"} and conversation_memory.get(
        "customer_configuration"
    ):
        return False
    return True


def extract_configuration(text):
    """Pull a BHK requirement from the current utterance. Returns a short label or None."""
    labels = {
        "1": "1 BHK", "one": "1 BHK",
        "2": "2 BHK", "two": "2 BHK",
        "3": "3 BHK", "three": "3 BHK",
        "4": "4 BHK", "four": "4 BHK",
    }
    match = re.search(
        r"\b([1-4]|one|two|three|four)\s*-?\s*(?:bhk|bedroom)",
        (text or "").lower(),
    )
    if match:
        return labels.get(match.group(1))
    last = (conversation_memory.get("last_agent_question") or "").lower()
    if not re.search(r"\b(configuration|bhk|bedroom)\b", last):
        return None
    low = (text or "").lower()
    if re.search(r"\b(minute|second|moment|wait|hour)\b", low):
        return None
    words = re.findall(r"[a-z0-9]+", low)
    if not words or len(words) > 4:
        return None
    numbers = [word for word in words if word in labels]
    if len(numbers) != 1:
        return None
    return labels[numbers[0]]


def extract_locality(text):
    """Pull a locality the customer named. Does not store the rest of the utterance."""
    text_lower = (text or "").lower()
    named = (
        (r"\bsouth\s+bengaluru\b", "South Bengaluru"),
        (r"\bsouth\s+bangalore\b", "South Bengaluru"),
        (r"\bwhitefield\b", "Whitefield"),
        (r"\belectronic city\b", "Electronic City"),
        (r"\bhosur road\b", "Hosur Road"),
    )
    for pattern, label in named:
        if re.search(pattern, text_lower):
            return label
    return None


def note_purpose(text):
    """Remember a purpose the customer actually stated."""
    low = (text or "").lower()
    if re.search(r"\binvestment\b", low):
        conversation_memory["customer_purpose"] = "investment"
    elif re.search(r"\b(own use|end use|to live|live in|for living)\b", low):
        conversation_memory["customer_purpose"] = "live"
    if conversation_memory.get("customer_purpose"):
        close_open_topic("purpose")


def close_open_topic(topic):
    topics = conversation_memory.setdefault("open_topics", [])
    if topic in topics:
        topics.remove(topic)


def remember_open(topic):
    topics = conversation_memory.setdefault("open_topics", [])
    if topic not in topics:
        topics.append(topic)
    if topic == "configuration":
        conversation_memory["configuration_asked"] = True


def remember_covered(topic):
    topics = conversation_memory.setdefault("covered_topics", [])
    if topic not in topics:
        topics.append(topic)


def remember_declined(topic):
    topics = conversation_memory.setdefault("declined_topics", [])
    if topic not in topics:
        topics.append(topic)


def note_customer_requirements(text):
    """Remember only the latest locality and configuration stated on this call."""
    configuration = extract_configuration(text)
    if configuration:
        conversation_memory["customer_configuration"] = configuration
        close_open_topic("configuration")
    note_variant(text)
    accept_offered_size(text)
    note_purpose(text)
    locality = extract_locality(text)
    if locality:
        conversation_memory["customer_locality"] = locality


CONFIGURATION_QUESTION = (
    "This is regarding SOBHA Townpark. "
    "We have one, two, three, and four BHK homes. "
    "Which configuration are you looking for?"
)
CONFIGURATION_FOLLOWUP = "One, two, three, or four BHK?"


def remember_spoken(text):
    """Keep the latest question only. This is not a transcript."""
    conversation_memory["last_spoken_reply"] = text
    questions = [part.strip() for part in re.findall(r"[^?]*\?", text or "") if part.strip()]
    question = questions[-1] if questions else None
    if question:
        pieces = split_sentences(question)
        question = pieces[-1] if pieces else question
    conversation_memory["last_agent_question"] = question
    if question and re.search(r"which configuration|one, two, three", question, re.I):
        conversation_memory["configuration_asked"] = True
    if re.search(r"\b(electronic city|hosur road)\b", text or "", re.I):
        remember_covered("location")
    if re.search(r"\bnew york\b", text or "", re.I):
        conversation_memory["theme_spoken"] = True
        remember_covered("home")
    if re.search(r"\bkind of home\b", text or "", re.I):
        remember_covered("home")
    if re.search(r"\bstarting price\b", text or "", re.I):
        remember_covered("price")
    if re.search(r"\bamenities\b", text or "", re.I):
        remember_covered("amenities")
    if re.search(
        r"\bwhat else (?:can i help you with|would you like to know)\b",
        text or "",
        re.I,
    ):
        conversation_memory["fallback_asked"] = True
    if re.search(r"\banything else about the project\b", text or "", re.I):
        conversation_memory["open_project_asked"] = True


def is_short_continuation(text):
    """Yes, okay, or a one-word reply to the question just asked."""
    cleaned = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
    if not cleaned or len(cleaned.split()) > 3 or is_clear_negative(cleaned):
        return False
    if cleaned in {
        "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "fine",
        "yes please", "okay sure", "ok sure",
    }:
        return True
    if cleaned in {"configuration", "configurations"}:
        last_question = (conversation_memory.get("last_agent_question") or "").lower()
        return re.search(r"\b(configuration|luxe|grande|layout)\b", last_question) is not None
    return False


def declining_last_offer(text):
    """No to the question just asked. A visit question is handled by the visit stages."""
    last_question = (conversation_memory.get("last_agent_question") or "").lower()
    if not last_question or re.search(r"\b(visit|schedule)\b", last_question):
        return False
    return is_clear_negative(text)


def answers_size_question(text):
    """A size or variant reply belongs to the layout question, not a new price question."""
    last_question = (conversation_memory.get("last_agent_question") or "").lower()
    if not re.search(r"\b(size|sizes|range|luxe|grande|sq)\b", last_question):
        return False
    normalized = (text or "").lower().replace("granded", "grande")
    return re.search(r"\b(luxe|grande|\d{3,4}|square|sq)\b", normalized) is not None


def note_variant(text):
    normalized = (text or "").lower().replace("granded", "grande")
    if re.search(r"\bgrande\b", normalized):
        conversation_memory["customer_variant"] = "Grande"
    elif re.search(r"\bluxe\b", normalized):
        conversation_memory["customer_variant"] = "Luxe"


def is_unsure(text):
    """Hesitation. This is not a yes and not a new size choice."""
    return re.search(
        r"\b(not sure|unsure|don'?t know|do not know|not certain|no idea)\b",
        (text or "").lower(),
    ) is not None


def is_flexible_about_size(text):
    """The customer will take whichever size is offered."""
    if is_unsure(text):
        return False
    return re.search(
        r"\b(anything|either|any size|whatever|flexible|doesn'?t matter|dont matter|"
        r"don'?t mind|do not mind|both are fine|either is fine|"
        r"fine with both|fine with either)\b",
        (text or "").lower(),
    ) is not None


def agrees_to_offered_size(text):
    """Yes to the size just spoken. A choice between two names is not a yes to one of them."""
    if is_unsure(text) or is_clear_negative(text) or is_flexible_about_size(text):
        return False
    if re.search(r"\b(i would like|tell me|what about|how about|know about)\b", (text or "").lower()):
        return False
    return is_positive(text)


def _variant_alone(text):
    low = (text or "").lower().replace("granded", "grande")
    has_luxe = re.search(r"\bluxe\b", low) is not None
    has_grande = re.search(r"\bgrande\b", low) is not None
    if has_grande and not has_luxe:
        return "Grande"
    if has_luxe and not has_grande:
        return "Luxe"
    return None


def larger_variant(config):
    """The bigger named option in the stored size line."""
    line = UNIT_SIZES.get(config) or ""
    found = re.findall(r"\b(Luxe|Grande)\b[^.]{0,40}?(\d{3,4})", line)
    if not found:
        return None
    return max(found, key=lambda item: int(item[1]))[0]


def variant_for_number(config, number):
    """Match a spoken square footage to Luxe or Grande."""
    line = UNIT_SIZES.get(config) or ""
    current = None
    for part in re.split(r"\b(Luxe|Grande)\b", line):
        if part in ("Luxe", "Grande"):
            current = part
        elif current and number in re.findall(r"\d{3,4}", part):
            return current
    return None


def single_offered_variant():
    """The one size named in the last reply. Both names means the customer still has to choose."""
    question = conversation_memory.get("last_agent_question") or ""
    reply = conversation_memory.get("last_spoken_reply") or ""
    for text in (question, reply):
        named = _variant_alone(text)
        if named:
            return named
    config = conversation_memory.get("customer_configuration")
    number = re.search(r"\b(\d{3,4})\b", question)
    if number:
        return variant_for_number(config, number.group(1))
    if re.search(r"\blarger\b", question, re.I):
        return conversation_memory.get("customer_variant") or larger_variant(config)
    return None


def size_topic_open():
    topic = " ".join(
        part for part in (
            conversation_memory.get("last_spoken_reply") or "",
            conversation_memory.get("last_agent_question") or "",
        ) if part
    )
    return re.search(r"\b(size|sizes|range|luxe|grande|sq|layout|square)\b", topic, re.I) is not None


def customer_reopens_size(text):
    """A new size question. Agreement, flexibility, and 'not sure' do not reopen it."""
    if is_unsure(text) or is_flexible_about_size(text) or agrees_to_offered_size(text):
        return False
    return re.search(
        r"\b(size|sizes|luxe|grande|sq|square|layout|how big)\b",
        (text or "").lower(),
    ) is not None


def accept_offered_size(text):
    """Store a size the customer accepted, including a yes to the size just offered."""
    if not size_topic_open() or is_unsure(text):
        return
    config = conversation_memory.get("customer_configuration")
    if is_flexible_about_size(text):
        reply = conversation_memory.get("last_spoken_reply") or ""
        both_named = re.search(r"\bluxe\b", reply, re.I) and re.search(r"\bgrande\b", reply, re.I)
        both_accepted = re.search(
            r"\b(fine with both|fine with either|both are fine|either is fine)\b",
            (text or "").lower(),
        )
        if both_named and both_accepted:
            return
        variant = _variant_alone(reply) or larger_variant(config)
        if variant:
            conversation_memory["customer_variant"] = variant
            conversation_memory["size_decided"] = True
        return
    if answers_size_question(text) and conversation_memory.get("customer_variant"):
        conversation_memory["size_decided"] = True
        return
    if agrees_to_offered_size(text):
        variant = single_offered_variant()
        if variant:
            conversation_memory["customer_variant"] = variant
            conversation_memory["size_decided"] = True


def reply_asks_about_size(answer):
    if "?" not in (answer or ""):
        return False
    return asks_customer_to_choose_size(answer)


def split_sentences(text):
    """Split on sentence ends. The periods inside 'sq. ft.' stay in the sentence."""
    protected = re.sub(
        r"\bsq\.",
        lambda match: match.group(0)[:-1] + "\u2024",
        text or "",
        flags=re.IGNORECASE,
    )
    protected = re.sub(
        r"\b(?i:ft)\.(?=\s+[a-z])",
        lambda match: match.group(0)[:-1] + "\u2024",
        protected,
    )
    parts = re.split(r"(?<=[.?!])\s+", protected)
    return [part.replace("\u2024", ".").strip() for part in parts if part.strip()]


def asks_customer_to_choose_size(sentence):
    """A question that asks for a size. Mentioning a range while asking something else is not one."""
    if "?" not in (sentence or ""):
        return False
    if re.search(
        r"\b(larger|smaller|compact|luxe or grande|which size|what size|"
        r"specific size|size range|within this|which of these sizes)\b",
        sentence,
        re.I,
    ):
        return True
    mentions_size = re.search(r"\b(sq\.?\s*ft|square feet|\d{3,4})\b", sentence, re.I)
    other_topic = re.search(
        r"\b(inventory|amenities|amenity|open space|location|payment|price|"
        r"visit|investment|live)\b",
        sentence,
        re.I,
    )
    return bool(mentions_size and not other_topic)


PURPOSE_QUESTION = "Are you looking at this to live in, or as an investment?"
LOCATION_QUESTION = "Would you like to know about the location?"
HOME_QUESTION = "Would you like to hear about the kind of home?"
PRICE_QUESTION = "Would you like to know the starting price?"
AMENITIES_QUESTION = "Would you like to hear about the amenities?"
OPEN_PROJECT_QUESTION = "Would you like to know anything else about the project?"
CALLBACK_QUESTION = (
    "That's completely fine. We do have other projects that can suit that. "
    "Can I arrange a call back from our sales representative?"
)


def blocked_topics():
    """Topics already spoken, declined, or left open. The next question must not use them."""
    blocked = set(conversation_memory.get("covered_topics") or [])
    blocked.update(conversation_memory.get("declined_topics") or [])
    blocked.update(conversation_memory.get("open_topics") or [])
    return blocked


def next_question():
    """One next question that is not the question just asked and not a parked topic."""
    last = conversation_memory.get("last_agent_question") or ""
    blocked = blocked_topics()
    choices = []
    if not conversation_memory.get("customer_purpose") and "purpose" not in blocked:
        choices.append(PURPOSE_QUESTION)
    if "location" not in blocked:
        choices.append(LOCATION_QUESTION)
    if "home" not in blocked:
        choices.append(HOME_QUESTION)
    if "price" not in blocked and "budget" not in blocked:
        choices.append(PRICE_QUESTION)
    if "amenities" not in blocked:
        choices.append(AMENITIES_QUESTION)
    for question in choices:
        if question != last:
            return question
    return "What else would you like to know?"


FALLBACK_QUESTION = "What else would you like to know?"
OFFER_THE_VISIT = object()


def question_after_purpose():
    """After a purpose answer, do not return to the location or the kind of home."""
    topics = conversation_memory.setdefault("covered_topics", [])
    added = [topic for topic in ("location", "home") if topic not in topics]
    topics.extend(added)
    try:
        return next_question()
    finally:
        for topic in added:
            if topic in topics:
                topics.remove(topic)


def follow_on(lead, question=None):
    """One next question. A second 'what else' offers the visit instead."""
    chosen = next_question() if question is None else question
    if (
        chosen == FALLBACK_QUESTION
        and conversation_memory.get("fallback_asked")
        and not conversation_memory.get("visit_offered")
    ):
        return OFFER_THE_VISIT
    return f"{lead} {chosen}".strip()


def reply_for_purpose(user_text):
    """An investment or own-use answer is acknowledged here. The model is not called."""
    if not is_purpose_statement(user_text):
        return None
    if re.search(
        r"\b(price|cost|how much|possession|amenities|configuration|how far|distance|offer)\b",
        user_text or "",
        re.I,
    ):
        return None
    note_purpose(user_text)
    if conversation_memory.get("customer_purpose") == "live":
        lead = "Got it. You're looking at this for your own use."
    else:
        lead = "Got it. You're looking at this as an investment."
    conversation_memory["stage"] = "answer_faq"
    if conversation_memory.get("price_stated") or "price" in (
        conversation_memory.get("covered_topics") or []
    ):
        conversation_memory["open_project_asked"] = True
        return f"{lead} {OPEN_PROJECT_QUESTION}"
    return f"{lead} {price_statement()}"


def price_statement():
    """The stored starting price, spoken once, with no permission question."""
    remember_covered("price")
    conversation_memory["price_stated"] = True
    return f"SOBHA Townpark starts at approximately {PROJECT_KB['starting_price']}."


def payment_and_open():
    """Payment, then one open question. Amenities are not offered here."""
    conversation_memory["payment_stated"] = True
    conversation_memory["open_project_asked"] = True
    return (
        f"{PROJECT_KB['payment']}. "
        "Our sales advisor can explain the current plans. "
        f"{OPEN_PROJECT_QUESTION}"
    )


def asks_about_payment(text):
    """They asked about payment. A no to that topic is not a request."""
    low = (text or "").lower()
    if not re.search(r"\bpayment\b", low):
        return False
    if re.search(r"\b(no|not|don't|dont|do not|without)\b", low):
        return False
    return True


def callback_offer():
    conversation_memory["awaiting_callback"] = True
    conversation_memory["stage"] = "answer_faq"
    return CALLBACK_QUESTION


def open_project_or_visit(lead):
    """Ask once if they want anything else. The next no offers the visit."""
    if conversation_memory.get("open_project_asked"):
        return OFFER_THE_VISIT
    conversation_memory["open_project_asked"] = True
    return f"{lead} {OPEN_PROJECT_QUESTION}".strip()


def budget_in_crore(text):
    low = (text or "").lower().replace(",", "")
    crore = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|cr)\b", low)
    if crore:
        return float(crore.group(1))
    lakh = re.search(r"(\d+(?:\.\d+)?)\s*lakh", low)
    if lakh:
        return float(lakh.group(1)) / 100.0
    return None


def price_is_too_high(text):
    """A budget at or below 1.8 crore, or a line that the price is too high."""
    if re.search(r"\btoo expensive\b", text or "", re.I):
        return True
    amount = budget_in_crore(text)
    return amount is not None and amount <= 1.8


def accepts_named_topic(user_text):
    """A longer yes still accepts the price or the payment that was just offered."""
    topic = topic_just_asked()
    low = (user_text or "").lower()
    if topic == "price" and re.search(r"\b(would like|want|like) to know\b", low):
        return True
    if topic == "payment" and re.search(r"\b(let me know|go ahead)\b", low):
        return True
    return False


def last_question_is_open_project():
    last = (conversation_memory.get("last_agent_question") or "").lower()
    return "anything else" in last and "project" in last


def reply_for_open_project(user_text):
    """A yes invites their question. A no offers the visit."""
    if not last_question_is_open_project():
        return None
    if is_direct_project_question(user_text) or is_new_faq_question(user_text):
        return None
    low = (user_text or "").lower()
    if (
        is_clear_negative(user_text)
        or declines_offered_topic(user_text)
        or is_unsure(user_text)
        or re.search(r"\b(nothing|no)\b", low)
    ):
        return OFFER_THE_VISIT
    if is_positive(user_text) or is_bare_okay(user_text):
        return "Sure, what would you like to know?"
    return None


def known_transition(user_text):
    """Price, payment, a lower budget, or the one open question. None leaves the turn alone."""
    if conversation_memory.get("awaiting_callback"):
        if is_direct_project_question(user_text) or is_new_faq_question(user_text):
            conversation_memory["awaiting_callback"] = False
            return None
        conversation_memory["awaiting_callback"] = False
        if is_clear_negative(user_text) or declines_offered_topic(user_text):
            return OFFER_THE_VISIT
        if is_positive(user_text):
            return "ADVISOR"
        return None
    if price_is_too_high(user_text):
        return callback_offer()
    if asks_about_payment(user_text):
        conversation_memory["stage"] = "answer_faq"
        return payment_and_open()
    if (
        conversation_memory.get("price_stated")
        and not conversation_memory.get("open_project_asked")
        and approves_the_price(user_text)
    ):
        conversation_memory["stage"] = "answer_faq"
        return open_project_or_visit("Understood.")
    return reply_for_open_project(user_text)


def route_known(reply):
    if reply == "ADVISOR":
        return advisor_followup_twiml()
    return response_for(reply)


def drop_restated_location(answer):
    """Once the greeting has given the location, do not say Electronic City again."""
    if "location" not in (conversation_memory.get("covered_topics") or []):
        return answer
    kept = [
        sentence
        for sentence in split_sentences(answer)
        if not re.search(r"\b(electronic city|hosur road)\b", sentence, re.I)
    ]
    return " ".join(kept)


def drop_restated_theme(answer):
    """Once the greeting has given the theme, do not say New York again."""
    if not conversation_memory.get("theme_spoken") or not answer:
        return answer
    kept = [
        sentence
        for sentence in split_sentences(answer)
        if not re.search(r"\bnew york\b", sentence, re.I)
    ]
    return " ".join(kept)


def asks_what_it_means(text):
    return re.search(r"\bwhat does (?:it|that|this) mean\b", text or "", re.I) is not None


def asks_for_a_floor_plan(text):
    """A layout question. A bare 'what does it mean' counts only after a layout line."""
    low = (text or "").lower()
    if re.search(
        r"\b(layout|floor\s*plan|floorplan|internal (?:zoning|joining)|zoning)\b",
        low,
    ):
        return True
    if not asks_what_it_means(text):
        return False
    last = " ".join(
        part for part in (
            conversation_memory.get("last_agent_question") or "",
            conversation_memory.get("last_spoken_reply") or "",
        ) if part
    ).lower()
    return re.search(r"\b(layout|floor\s*plan|floorplan|zoning|joining)\b", last) is not None


def drop_invented_floor_plan(answer):
    """Drop a model question about a layout, a floor plan, or zoning."""
    kept = []
    for sentence in split_sentences(answer or ""):
        if "?" in sentence and re.search(
            r"\b(layout|floor\s*plan|floorplan|zoning|joining)\b",
            sentence,
            re.I,
        ):
            continue
        if re.search(r"\b(internal zoning|internal joining|zoning)\b", sentence, re.I):
            continue
        kept.append(sentence)
    return " ".join(kept)


def reply_for_floor_plan(user_text):
    """The stored size is the floor-plan fact. The model does not invent one."""
    if not asks_for_a_floor_plan(user_text):
        return None
    parts = []
    config = conversation_memory.get("customer_configuration")
    size = UNIT_SIZES.get(config or "")
    last = conversation_memory.get("last_spoken_reply") or ""
    numbers = re.findall(r"\d{3,4}", size or "")
    if size and (not numbers or not all(number in last for number in numbers)):
        parts.append(size.rstrip("."))
    parts.append("I don't have a floor plan beyond that size.")
    conversation_memory["stage"] = "answer_faq"
    return follow_on(" ".join(parts))


def move_on_line():
    """One next question. A second 'what else' offers the visit instead."""
    return follow_on("That's alright.")


def topic_of_text(text):
    """The fact a line is about, if it is one of these."""
    last = (text or "").lower()
    if not last:
        return None
    if re.search(r"\bbudget\b", last):
        return "budget"
    if re.search(r"\bpayment\b", last):
        return "payment"
    if re.search(r"\b(price|cost|spend|crore)\b", last):
        return "price"
    if re.search(r"\b(amenities|amenity)\b", last):
        return "amenities"
    if re.search(r"\blocation\b", last):
        return "location"
    if re.search(r"\bkind of home\b", last):
        return "home"
    if re.search(
        r"\b(size|sizes|range|luxe|grande|sq|larger|smaller|layout|square)\b",
        last,
    ):
        return "size"
    if re.search(r"\b(configurations?|bhk|bedroom)\b", last):
        return "configuration"
    if re.search(r"\b(investment|live in|own use)\b", last):
        return "purpose"
    if re.search(r"\b(visit|schedule)\b", last):
        return "visit"
    return None


def fact_for_offered_topic(topic):
    """The stored project fact for a topic the customer agreed to hear."""
    if topic == "home":
        return "SOBHA Townpark is a New York-themed luxury community."
    if topic == "price":
        return f"SOBHA Townpark starts at approximately {PROJECT_KB['starting_price']}."
    if topic == "payment":
        return PROJECT_KB["payment"] + "."
    if topic == "amenities":
        return f"SOBHA Townpark offers {PROJECT_KB['amenities']}."
    if topic == "location":
        return f"SOBHA Townpark is located {PROJECT_KB['location']}."
    return None


def reply_for_accepted_topic(user_text):
    """A yes to an offered topic is answered here. A yes to a visit is not."""
    heard = is_positive(user_text) or asks_what_it_means(user_text) or accepts_named_topic(user_text)
    if not heard or not conversation_memory.get("last_agent_question"):
        return None
    topic = topic_just_asked()
    if topic == "home" and conversation_memory.get("theme_spoken"):
        conversation_memory["stage"] = "answer_faq"
        conversation_memory["awaiting_visit_decision"] = False
        return follow_on("Understood.")
    if topic == "price":
        conversation_memory["stage"] = "answer_faq"
        conversation_memory["awaiting_visit_decision"] = False
        return price_statement()
    if topic == "payment":
        conversation_memory["stage"] = "answer_faq"
        conversation_memory["awaiting_visit_decision"] = False
        return payment_and_open()
    fact = fact_for_offered_topic(topic)
    if not fact:
        return None
    remember_covered(topic)
    conversation_memory["stage"] = "answer_faq"
    conversation_memory["awaiting_visit_decision"] = False
    return follow_on(fact)


def topic_just_asked():
    """The fact the last question was waiting for, if it was one of these."""
    return topic_of_text(conversation_memory.get("last_agent_question") or "")


def leaves_any_configuration(text):
    """Any configuration is an open answer. It does not choose Luxe or Grande."""
    return re.search(
        r"\bany configuration\b|\bokay with any\b|\bfine with any\b",
        (text or "").lower(),
    ) is not None


def asks_to_hear_about_the_project(text):
    """They asked to hear about the project, not about one size."""
    return re.search(
        r"\b(?:about|regarding) the project\b",
        (text or "").lower(),
    ) is not None


def leaves_fact_open(text):
    """The customer is not supplying the fact just asked."""
    if is_unsure(text) or is_flexible_about_size(text) or leaves_any_configuration(text):
        return True
    return re.search(
        r"\b(no preference|not decided|haven'?t decided|haven'?t finalized|"
        r"not finalized|yet to finalize|you tell me|you decide|"
        r"decide(?:\s+this)?\s+later|"
        r"skip that|whatever you (?:think|suggest)|either way)\b",
        (text or "").lower(),
    ) is not None


def acknowledge_open(topic):
    """One acknowledgement and one different question."""
    if topic == "budget":
        nxt = next_question() if "location" in blocked_topics() else (
            HOME_QUESTION
            if (conversation_memory.get("last_agent_question") or "") == LOCATION_QUESTION
            else LOCATION_QUESTION
        )
        return f"That's alright, we can leave the budget for now. {nxt}"
    if topic == "size" and conversation_memory.get("size_decided"):
        variant = conversation_memory.get("customer_variant") or "selected"
        config = conversation_memory.get("customer_configuration") or "home"
        lead = f"That's alright, we'll keep the {variant} {config} range."
    elif topic == "size":
        lead = "That's alright, we can leave the size open."
    elif topic == "configuration":
        lead = "That's alright, we can leave the configuration open."
    else:
        lead = "That's alright."
    return f"{lead} {next_question()}"


def reply_for_open_answer(user_text):
    """Acknowledge an open answer before the model writes a reply. None when the model should answer."""
    if leaves_any_configuration(user_text):
        remember_open("configuration")
        lead = "That's alright, we can leave the configuration open."
        if asks_to_hear_about_the_project(user_text):
            conversation_memory["stage"] = "answer_faq"
            if conversation_memory.get("price_stated"):
                return f"{lead} {OPEN_PROJECT_QUESTION}"
            return f"{lead} {price_statement()}"
        return acknowledge_open("configuration")
    if not leaves_fact_open(user_text) or not conversation_memory.get("last_agent_question"):
        return None
    topic = topic_just_asked()
    if not topic and size_topic_open():
        topic = "size"
    if topic == "size":
        if conversation_memory.get("size_decided"):
            conversation_memory["size_closed_before_turn"] = True
            close_open_topic("size")
        else:
            remember_open("size")
        return acknowledge_open("size")
    if topic:
        remember_open(topic)
        return acknowledge_open(topic)
    if is_unsure(user_text) or is_flexible_about_size(user_text):
        return move_on_line()
    return None


def leave_closed_size(user_query, answer):
    """Once a size is stored, drop a later size question and keep the rest of the answer."""
    if not (
        conversation_memory.get("size_decided")
        and conversation_memory.get("size_closed_before_turn")
        and not customer_reopens_size(user_query)
    ):
        return answer
    kept = [
        sentence
        for sentence in split_sentences(answer)
        if not asks_customer_to_choose_size(sentence)
    ]
    if not kept:
        return ""
    return " ".join(kept)


def customer_asks_about_configuration(text):
    """The customer asked to hear configurations. A parked topic stays parked otherwise."""
    return re.search(
        r"\b(configuration|configurations|floor plan|floorplan|bhk|bedroom|layout|sizes|size)\b",
        text or "",
        re.I,
    ) is not None


def sentence_asks_to_pick_configuration(sentence):
    """Any question that brings configurations back, including 'available configurations'."""
    if "?" not in (sentence or ""):
        return False
    return re.search(
        r"\b(configurations?|which bhk|specific configuration|specific sizes|"
        r"available configuration|available sizes|"
        r"one, two, three|smaller or larger|larger or smaller)\b",
        sentence,
        re.I,
    ) is not None


def sentence_lists_bhk_menu(sentence):
    """A recital of the 1 to 4 BHK menu, not one home's size."""
    if re.search(r"\b1\s*bhk\s*to\s*4\s*bhk\b", sentence or "", re.I):
        return True
    if re.search(r"\b(1|one)\s+to\s+(4|four)\s+bhk\b", sentence or "", re.I):
        return True
    if re.search(r"\b(one|1)\s+to\s+(four|4)\s+bedrooms?\b", sentence or "", re.I):
        return True
    if re.search(
        r"\b(1|one)\b.{0,40}\b(2|two)\b.{0,40}\b(3|three)\b.{0,40}\b(4|four)\b",
        sentence or "",
        re.I,
    ):
        return True
    labels = []
    for item in re.findall(r"\b[1-4]\s*bhk\b", sentence or "", re.I):
        key = re.sub(r"\s+", " ", item.lower())
        if key not in labels:
            labels.append(key)
    return len(labels) >= 3


_PERCENT_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_PERCENT_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_int(phrase):
    """A percent spoken as words, such as sixty seven. None when it is not a number."""
    parts = re.split(r"[-\s]+", (phrase or "").lower().strip())
    if len(parts) == 1 and parts[0] in _PERCENT_ONES:
        return _PERCENT_ONES[parts[0]]
    if len(parts) == 1 and parts[0] in _PERCENT_TENS:
        return _PERCENT_TENS[parts[0]]
    if len(parts) == 2 and parts[0] in _PERCENT_TENS and parts[1] in _PERCENT_ONES:
        return _PERCENT_TENS[parts[0]] + _PERCENT_ONES[parts[1]]
    return None


def _spoken_percent_numbers(sentence):
    """The percents in one sentence, as digit strings."""
    numbers = []
    for match in re.finditer(r"\d+\s*(?:%|percent)", sentence or "", re.I):
        digits = re.sub(r"\D", "", match.group(0))
        if digits:
            numbers.append(digits)
    ones = "|".join(_PERCENT_ONES)
    tens = "|".join(_PERCENT_TENS)
    pattern = rf"\b((?:{tens})(?:[-\s](?:{ones}))?|(?:{ones}))\s+percent\b"
    for match in re.finditer(pattern, sentence or "", re.I):
        number = _words_to_int(match.group(1))
        if number is not None:
            numbers.append(str(number))
    return numbers


def _percent_in_source(number, source):
    compact = re.sub(r"\s+", "", (source or "").lower())
    return f"{number}%" in compact or f"{number}percent" in compact


def drop_unsupported_percent(answer, source=None):
    """Drop a percent the project facts and the retrieved brochure passage do not contain."""
    facts = source if source is not None else " ".join(
        str(value) for value in PROJECT_KB.values()
    )
    kept = []
    for sentence in split_sentences(answer):
        numbers = _spoken_percent_numbers(sentence)
        if any(not _percent_in_source(number, facts) for number in numbers):
            continue
        kept.append(sentence)
    return " ".join(kept)


def asks_about_an_amenity(text):
    """Open space, the clubhouse, the pool, or the amenities list."""
    return re.search(
        r"\b(amenities|amenity|open spaces?|club\s*house|swimming pools?|pools?)\b",
        text or "",
        re.I,
    ) is not None


def drop_unasked_amenity_question(user_query, answer):
    """Do not offer an amenity. A question about one is answered, not followed by another."""
    asked = asks_about_an_amenity(user_query) or (
        asks_about_an_amenity(conversation_memory.get("last_agent_question") or "")
        and (is_positive(user_query) or asks_what_it_means(user_query))
    )
    kept = []
    removed = False
    for sentence in split_sentences(answer or ""):
        if "?" in sentence and asks_about_an_amenity(sentence):
            removed = True
            continue
        kept.append(sentence)
    spoken = " ".join(kept)
    if not spoken and removed:
        conversation_memory["open_project_asked"] = True
        return OPEN_PROJECT_QUESTION
    if removed and spoken and "?" not in spoken and not conversation_memory.get("open_project_asked"):
        conversation_memory["open_project_asked"] = True
        return f"{spoken} {OPEN_PROJECT_QUESTION}"
    if asked and spoken and "?" not in spoken and not conversation_memory.get("open_project_asked"):
        conversation_memory["open_project_asked"] = True
        return f"{spoken} {OPEN_PROJECT_QUESTION}"
    return spoken


def leave_closed_configuration(user_query, answer):
    """Once configuration is left open, do not ask the BHK menu again unless they ask."""
    if conversation_memory.get("customer_configuration"):
        return answer
    if "configuration" not in (conversation_memory.get("open_topics") or []):
        return answer
    asked = customer_asks_about_configuration(user_query)
    kept = []
    for sentence in split_sentences(answer):
        if sentence_asks_to_pick_configuration(sentence):
            continue
        if not asked and sentence_lists_bhk_menu(sentence):
            continue
        kept.append(sentence)
    if not kept:
        return ""
    spoken = " ".join(kept)
    if "?" not in spoken:
        return f"{spoken} {next_question()}"
    return spoken


def sentence_claims_chosen_configuration(sentence):
    """A claim that a configuration was chosen, when none was stored."""
    return re.search(
        r"\b(already chosen|have chosen|you chose|you selected)\b",
        sentence or "",
        re.I,
    ) is not None and re.search(r"\bconfiguration\b", sentence or "", re.I) is not None


def drop_false_configuration_claim(answer):
    if not answer or conversation_memory.get("customer_configuration"):
        return answer
    kept = [
        sentence
        for sentence in split_sentences(answer)
        if not sentence_claims_chosen_configuration(sentence)
    ]
    return " ".join(kept)


def leave_blocked_topics(answer):
    """Drop a question whose topic was already spoken or declined."""
    if not answer:
        return answer
    blocked = blocked_topics()
    kept = []
    dropped = False
    for sentence in split_sentences(answer):
        topic = topic_of_text(sentence)
        if topic and topic in blocked and "?" in sentence:
            dropped = True
            continue
        kept.append(sentence)
    if not kept:
        return ""
    spoken = " ".join(kept)
    if dropped and "?" not in spoken:
        return f"{spoken} {next_question()}"
    return spoken


def drop_question_just_answered(answer):
    """Do not ask to hear a topic the same reply has just stated."""
    if not answer:
        return answer
    stated = set()
    kept = []
    dropped = False
    for sentence in split_sentences(answer):
        topic = topic_of_text(sentence)
        if topic in ("home", "price", "amenities", "location") and "?" not in sentence:
            stated.add(topic)
            remember_covered(topic)
        if topic in stated and "?" in sentence:
            dropped = True
            continue
        kept.append(sentence)
    if not kept:
        return ""
    spoken = " ".join(kept)
    if dropped and "?" not in spoken:
        return f"{spoken} {next_question()}"
    return spoken


def declines_offered_topic(text):
    """A no to the last offer. 'Not interested in that' is the offer, not the whole call."""
    low = (text or "").lower()
    if re.search(r"\bnot interested in (that|it|those|them)\b", low):
        return True
    if re.search(r"\bnot interested to know\b", low):
        return True
    if re.search(r"\bno,?\s+i am not\b", low):
        return True
    return re.search(
        r"\b(already told|already said|already asked|already answered|"
        r"you told|you mentioned|you asked|heard that|didn'?t answer|repeating)\b",
        low,
    ) is not None


def declines_the_project(text):
    """A bare 'not interested' ends the call. 'Not interested in that' does not."""
    if declines_offered_topic(text):
        return False
    return is_not_interested(text)


def reply_for_declined_topic(user_text):
    """Acknowledge a no to the last topic and ask about something still open."""
    if not declines_offered_topic(user_text) or not conversation_memory.get("last_agent_question"):
        return None
    topic = topic_just_asked()
    if topic:
        remember_declined(topic)
    if re.search(r"\b(already told|already said|you told|you mentioned)\b", user_text or "", re.I):
        lead = "Understood."
    else:
        lead = "That's alright."
    conversation_memory["stage"] = "answer_faq"
    conversation_memory["awaiting_visit_decision"] = False
    if topic == "amenities" or re.search(r"\bnot interested to know\b", user_text or "", re.I):
        return open_project_or_visit(lead)
    return follow_on(lead)


def _variant_number_map(label):
    line = UNIT_SIZES.get(label) or ""
    chunks = re.split(r"\b(Luxe|Grande)\b", line)
    mapping = {"Luxe": [], "Grande": []}
    current = None
    for chunk in chunks:
        if chunk in ("Luxe", "Grande"):
            current = chunk
        elif current:
            mapping[current].extend(re.findall(r"\d{3,4}", chunk))
    return mapping


def _named_configs(sentence):
    found = []
    for number, label in (("1", "1 BHK"), ("2", "2 BHK"), ("3", "3 BHK"), ("4", "4 BHK")):
        if re.search(rf"\b{number}\s*bhk\b", sentence or "", re.I) and label not in found:
            found.append(label)
    return found


def _segment_matches(label, part):
    numbers = re.findall(r"\d{3,4}", part or "")
    allowed = re.findall(r"\d{3,4}", UNIT_SIZES.get(label) or "")
    if not numbers or any(number not in allowed for number in numbers):
        return False
    variant_numbers = _variant_number_map(label)
    for variant in ("Luxe", "Grande"):
        claimed = []
        for match in re.finditer(
            rf"(\d{{3,4}})(?:\s+to\s+(\d{{3,4}}))?(?:\s+sq\.?\s*ft\.?)?\s+in\s+{variant}\b",
            part or "",
            re.I,
        ):
            claimed.append(match.group(1))
            if match.group(2):
                claimed.append(match.group(2))
        for match in re.finditer(
            rf"\b{variant}\b(?:\s+\w+){{0,6}}\s+(\d{{3,4}})(?:\s+to\s+(\d{{3,4}}))?",
            part or "",
            re.I,
        ):
            claimed.append(match.group(1))
            if match.group(2):
                claimed.append(match.group(2))
        if claimed and any(number not in variant_numbers.get(variant, []) for number in claimed):
            return False
    return True


def _recites_a_size(sentence):
    return (
        re.search(r"\b\d{3,4}\b", sentence or "") is not None
        and re.search(r"\b(sq|square|bhk|luxe|grande)\b", sentence or "", re.I) is not None
    )


def use_unit_sizes(user_query, answer):
    """A size recital uses the stored ranges. Swapped Luxe and Grande labels are replaced."""
    if not answer:
        return answer
    stored = conversation_memory.get("customer_configuration")
    asked = customer_asks_about_configuration(user_query)
    rebuilt = []
    for sentence in split_sentences(answer):
        if not _recites_a_size(sentence):
            rebuilt.append(sentence)
            continue
        named = _named_configs(sentence)
        if stored in UNIT_SIZES and not asked and len(named) > 1:
            rebuilt.append(UNIT_SIZES[stored])
            continue
        if not named and stored in UNIT_SIZES:
            named = [stored]
        if named and any(not _segment_matches(label, sentence) for label in named):
            rebuilt.append(" ".join(UNIT_SIZES[label] for label in named))
            continue
        rebuilt.append(sentence)
    return " ".join(rebuilt)


def restore_denied_sizes(answer):
    """A stored size is not 'not listed'. Keep that line and ask the open question."""
    config = conversation_memory.get("customer_configuration")
    stored = UNIT_SIZES.get(config or "")
    if not stored or not answer:
        return answer
    sentences = split_sentences(answer)
    denied = False
    kept = []
    for sentence in sentences:
        denies = re.search(
            r"\b(not listed|not available|are not here|is not here)\b",
            sentence,
            re.I,
        ) and re.search(r"\b(size|sizes|sq|bhk|luxe|grande)\b", sentence, re.I)
        if denies:
            denied = True
            continue
        if re.search(r"\b(share|details)\b", sentence, re.I):
            kept.append(sentence)
            continue
        kept.append(sentence)
    if not denied:
        return answer
    kept = [
        sentence for sentence in kept
        if not re.search(r"\b(share|details)\b", sentence, re.I)
    ]
    conversation_memory["open_project_asked"] = True
    body = " ".join(kept).strip()
    if body:
        return f"{body} {stored} {OPEN_PROJECT_QUESTION}"
    return f"{stored} {OPEN_PROJECT_QUESTION}"


def drop_unasked_payment(user_query, answer):
    """A size choice should not be turned into a payment pitch."""
    if not answers_size_question(user_query):
        return answer
    if re.search(r"\b(price|payment|cost|crore)\b", (user_query or "").lower()):
        return answer
    sentences = re.findall(r"[^.?!]+[.?!]?", answer or "")
    kept = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() and not re.search(r"\b(payment|price|crore)\b", sentence, re.I)
    ]
    if kept:
        return " ".join(kept)
    variant = conversation_memory.get("customer_variant") or "selected"
    config = conversation_memory.get("customer_configuration") or "home"
    return f"That's the {variant} {config}. What else would you like to know about it?"


def decline_followup():
    """Leave the refused topic and ask what the customer wants next."""
    last_question = (conversation_memory.get("last_agent_question") or "").lower()
    topics = conversation_memory.setdefault("declined_topics", [])
    if "payment" in last_question and "payment" not in topics:
        topics.append("payment")
    elif re.search(r"\b(price|pricing)\b", last_question) and "pricing" not in topics:
        topics.append("pricing")
    if re.search(r"what would you like to know", last_question):
        return None
    config = conversation_memory.get("customer_configuration") or "home"
    variant = conversation_memory.get("customer_variant")
    home = f"{variant} {config}".strip() if variant else config
    if "payment" in topics:
        return f"Alright, we'll leave the payment plans. What would you like to know about the {home}?"
    if "pricing" in topics:
        return "Understood. What else would you like to know about the home?"
    return "Alright. What would you like to know next?"


def answering_last_question(text):
    if not conversation_memory.get("last_agent_question"):
        return False
    return is_short_continuation(text) or declining_last_offer(text) or answers_size_question(text)


def turn_intent(user_text):
    """A short reply keeps the topic of the question just asked."""
    if asks_about_an_amenity(user_text):
        return "amenities"
    last = conversation_memory.get("last_agent_question") or ""
    if asks_about_an_amenity(last) and (
        is_positive(user_text) or asks_what_it_means(user_text)
    ):
        return "amenities"
    if answering_last_question(user_text) and conversation_memory.get("last_intent"):
        return conversation_memory["last_intent"]
    return detect_intent(user_text)


def turn_query(user_text):
    if answering_last_question(user_text):
        return f"{conversation_memory.get('last_agent_question')} {user_text}"
    return user_text


def is_location_concern(text):
    """Customer commute or area preference. This is not the project's own location."""
    text_lower = (text or "").lower()
    if re.search(
        r"\b(too far|how far|closer|distance|commute|convenient|workplace|orr)\b",
        text_lower,
    ):
        return True
    if re.search(r"\b(working in|live in|living in|looking around|closer to)\b", text_lower):
        return True
    if extract_locality(text) and re.search(r"\b(working|around|far|closer|from)\b", text_lower):
        return True
    return False


def missing_requirement_question(text):
    """One qualification question, or None when the customer's point should be answered."""
    if (
        is_direct_project_question(text)
        or is_purpose_statement(text)
        or is_location_concern(text)
        or wants_site_visit(text)
        or needs_time(text)
    ):
        return None
    if not conversation_memory.get("customer_configuration"):
        if "configuration" in (conversation_memory.get("open_topics") or []):
            return None
        if conversation_memory.get("configuration_asked"):
            if is_short_continuation(text):
                return CONFIGURATION_FOLLOWUP
            return None
        return CONFIGURATION_QUESTION
    return None


def qualification_twiml(question):
    """Ask one qualification question and keep listening."""
    return f"<Response>{gather_block(question)}</Response>"


def time_to_decide_twiml():
    """Accept hesitation and offer details. Does not start visit booking."""
    conversation_memory["stage"] = "closed"
    return Response(
        content=polite_exit_twiml(
            "No problem, take your time. "
            "I can ask our team to share the available project details with you. "
            "Thank you, and have a good day."
        ),
        media_type="application/xml",
    )


def escape_twiml(text):
    """Escape special characters for safe Twilio XML output."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def reset_memory():
    """Reset per-call conversation state."""
    conversation_memory["stage"] = "greeting"
    conversation_memory["faq_count"] = 0
    conversation_memory["last_intent"] = None
    conversation_memory["visit_preference"] = None
    conversation_memory["awaiting_visit_decision"] = False
    conversation_memory["visit_day"] = None
    conversation_memory["selected_slot"] = None
    conversation_memory["empty_retry_count"] = 0
    conversation_memory["slot_pick_retries"] = 0
    conversation_memory["customer_locality"] = None
    conversation_memory["customer_configuration"] = None
    conversation_memory["last_agent_question"] = None
    conversation_memory["configuration_asked"] = False
    conversation_memory["customer_variant"] = None
    conversation_memory["size_decided"] = False
    conversation_memory["size_closed_before_turn"] = False
    conversation_memory["last_spoken_reply"] = None
    conversation_memory["declined_topics"] = []
    conversation_memory["covered_topics"] = []
    conversation_memory["open_topics"] = []
    conversation_memory["customer_purpose"] = None
    conversation_memory["visit_offered"] = False
    conversation_memory["fallback_asked"] = False
    conversation_memory["theme_spoken"] = False
    conversation_memory["price_stated"] = False
    conversation_memory["payment_stated"] = False
    conversation_memory["open_project_asked"] = False
    conversation_memory["awaiting_callback"] = False


def grounded_reply(user_text):
    """One spoken answer from the customer's words. Does not change the stage."""
    note_customer_requirements(user_text)
    intent = turn_intent(user_text)
    conversation_memory["last_intent"] = intent
    brochure_context, top_score = retrieve_context(intent, turn_query(user_text))
    return generate_response(user_text, intent, brochure_context, top_score)


def answer_in_place(user_text):
    """Speak that answer and keep listening on the current stage."""
    return Response(
        content=f"<Response>{gather_block(grounded_reply(user_text))}</Response>",
        media_type="application/xml",
    )


def answer_then_ask(user_text, question):
    """Answer what was said, then ask the stage question again."""
    return Response(
        content=f"""
<Response>

{say_block(grounded_reply(user_text))}

<Pause length="1"/>

{gather_block(question)}

</Response>
""",
        media_type="application/xml",
    )


def answer_faq_and_prompt(user_text, nudge_visit=False):
    """Retrieve context and speak one grounded reply. Visit booking stays outside this text."""
    answer = grounded_reply(user_text)
    conversation_memory["faq_count"] += 1

    if nudge_visit:
        conversation_memory["awaiting_visit_decision"] = True
        return f"""
<Response>

{say_block(answer)}

<Pause length="1"/>

{gather_block(visit_schedule_question())}

</Response>
"""

    return f"<Response>{gather_block(answer)}</Response>"


def propose_slot_twiml():
    """Ask once whether to schedule a visit."""
    conversation_memory["visit_offered"] = True
    return f"""
<Response>

{gather_block(visit_schedule_question())}

</Response>
"""


def visit_day_twiml():
    """After yes to visit — ask which day only (time comes next as fixed slots)."""
    conversation_memory["visit_offered"] = True
    return f"""
<Response>

{gather_block(
    "Great. Which day would suit you best for the visit — for example today, tomorrow, "
    "this Saturday or Sunday, or a weekday?"
)}

</Response>
"""


def visit_slot_choice_twiml(is_retry=False):
    """Present the four fixed slots."""
    prefix = (
        "Sorry, I didn't catch the time. "
        if is_retry
        else ""
    )
    prompt = (
        f"{prefix}"
        f"We have these slots — {VISIT_SLOTS_SPOKEN}. "
        "Which one works best for you?"
    )
    return f"""
<Response>

{gather_block(prompt)}

</Response>
"""


def booking_confirmed_twiml():
    """Final confirmation after day + slot captured."""
    day = conversation_memory.get("visit_day") or "your chosen day"
    slot = conversation_memory.get("selected_slot") or ""
    safe_day = escape_twiml(day)
    safe_slot = escape_twiml(slot)
    msg = (
        f"Wonderful. Your slot has been booked for {safe_day}, at {safe_slot}. "
        "Our sales advisor will reconfirm the visit with you. "
        "Thank you for your time. Have a great day."
    )
    return f"""
<Response>
{say_block(msg, escape_content=False)}
<Hangup/>
</Response>
"""


def polite_exit_twiml(message):
    """Say message and hang up using Indian voice."""
    return f"""
<Response>
{say_block(message)}
<Hangup/>
</Response>
"""


def empty_speech_fallback(stage):
    """Reprompt when Twilio sends empty SpeechResult (noise / timeout)."""
    conversation_memory["empty_retry_count"] = conversation_memory.get("empty_retry_count", 0) + 1
    count = conversation_memory["empty_retry_count"]
    if count >= MAX_EMPTY_RETRIES:
        conversation_memory["stage"] = "closed"
        return Response(
            content=polite_exit_twiml(
                "I'm having a little trouble hearing you clearly. "
                "I'll arrange for our sales advisor to reach out on this number. "
                "Thank you for your time. Have a great day."
            ),
            media_type="application/xml",
        )
    sorry = "Sorry, I couldn't hear you clearly. "
    if stage == "greeting":
        prompt = sorry + (
            "Is this a good time to chat for just a minute about SOBHA Townpark?"
        )
    elif stage == "qualify":
        if not conversation_memory.get("customer_configuration"):
            prompt = sorry + (
                CONFIGURATION_FOLLOWUP
                if conversation_memory.get("configuration_asked")
                else CONFIGURATION_QUESTION
            )
        else:
            prompt = sorry + "Could you please repeat that in a few simple words?"
    elif stage == "visit_day":
        prompt = sorry + (
            "Which day works best for your visit — today, tomorrow, weekend, or a weekday?"
        )
    elif stage == "visit_pick_slot":
        prompt = sorry + (
            f"Which slot would you prefer — {VISIT_SLOTS_SPOKEN}?"
        )
    elif stage == "propose_slot":
        prompt = sorry + visit_schedule_question()
    elif stage == "answer_faq" and conversation_memory.get("awaiting_visit_decision"):
        prompt = sorry + visit_schedule_question()
    else:
        prompt = sorry + (
            "Could you please repeat that in a few simple words?"
        )
    return Response(
        content=f"<Response>{gather_block(prompt)}</Response>",
        media_type="application/xml",
    )


# =====================================================
# START CALL
# =====================================================


@app.api_route("/voice", methods=["GET", "POST"])
async def voice():
    """Start the call and begin the greeting stage (GET or POST — Twilio may use either)."""
    try:
        reset_memory()

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>

{gather_block(
    "Hi, this is Sobha Limited calling. I'm getting in touch about SOBHA Townpark - "
    "a New York-inspired luxury community near Electronic City on Hosur Road in Bengaluru. "
    "Is now a good time for a quick call?"
)}

</Response>
"""
        return Response(content=twiml, media_type="application/xml")
    except Exception:
        traceback.print_exc()
        fallback = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
<Say>An application error occurred. Goodbye.</Say>
<Hangup/>
</Response>
"""
        return Response(content=fallback, media_type="application/xml")


def decline_project_twiml():
    """A customer who is not interested ends the call. This is not a visit offer."""
    return advisor_followup_twiml()


def advisor_followup_twiml():
    """One polite close after a project no, a deferred visit, or no details."""
    conversation_memory["stage"] = "closed"
    return Response(
        content=polite_exit_twiml(ADVISOR_CLOSE),
        media_type="application/xml",
    )


def share_details_twiml():
    """They agreed to documents or highlights. Arrange that, then stop."""
    conversation_memory["stage"] = "closed"
    return Response(
        content=polite_exit_twiml(SHARE_CLOSE),
        media_type="application/xml",
    )


def response_for(reply):
    """Speak a known line, or offer the visit when that line would repeat the fallback."""
    if reply is OFFER_THE_VISIT:
        return offer_the_visit_once()
    if reply:
        return Response(content=qualification_twiml(reply), media_type="application/xml")
    return None


def offer_the_visit_once():
    """A no to one topic can still hear the visit question one time."""
    conversation_memory["visit_offered"] = True
    conversation_memory["stage"] = "propose_slot"
    return Response(content=propose_slot_twiml(), media_type="application/xml")


def open_answer_response(user_text, from_qualify=False):
    """Speak an open answer. A second fallback offers the visit instead."""
    opened = reply_for_open_answer(user_text)
    if opened is OFFER_THE_VISIT:
        return offer_the_visit_once()
    if not opened:
        return None
    if from_qualify:
        conversation_memory["stage"] = "answer_faq"
    return Response(content=qualification_twiml(opened), media_type="application/xml")


def close_or_offer_the_visit():
    """Offer the visit once. The same goodbye after that visit closes the call."""
    stage = conversation_memory.get("stage")
    if conversation_memory.get("visit_offered") or stage in (
        "propose_slot", "visit_day", "visit_pick_slot",
    ):
        return advisor_followup_twiml()
    return offer_the_visit_once()


def polite_stop(user_text):
    """Close when the customer is done with the project, the visit, or any details."""
    if declines_the_project(user_text) or refuses_details(user_text) or asks_to_call_later(user_text):
        return advisor_followup_twiml()
    on_visit = conversation_memory.get("stage") in (
        "propose_slot", "visit_day", "visit_pick_slot",
    )
    if on_visit and (
        defers_the_visit(user_text)
        or has_not_finalized(user_text)
        or finished_this_topic(user_text)
        or is_clear_negative(user_text)
    ):
        return advisor_followup_twiml()
    return None


def yes_to_materials(user_text):
    """A yes to highlights or documents arranges the details. It does not book a day."""
    if not is_positive(user_text) or wants_site_visit(user_text):
        return False
    if last_question_is_visit():
        return False
    return last_question_offers_materials() or agrees_to_materials(user_text)


def visit_needs_a_clear_yes_twiml():
    """A bare okay does not book the visit."""
    conversation_memory["stage"] = "propose_slot"
    conversation_memory["awaiting_visit_decision"] = False
    return Response(
        content=qualification_twiml(
            "That's alright. Would you like to schedule a site visit, or shall I leave it for now?"
        ),
        media_type="application/xml",
    )


# =====================================================
# HANDLE SPEECH
# =====================================================


@app.post("/handle-speech")
async def handle_speech(request: Request, SpeechResult: str = Form(default="")):
    """Greeting → Qualify → Answer FAQ → Propose visit → Day → Pick slot → Booked → Closed"""
    form = await request.form()
    raw_text = (SpeechResult or "").strip()
    if not raw_text:
        raw_text = (form.get("UnstableSpeechResult") or "").strip()
    stage = conversation_memory.get("stage", "greeting")

    print("TWILIO FORM:", dict(form))

    if not raw_text:
        return empty_speech_fallback(stage)

    conversation_memory["empty_retry_count"] = 0
    user_text = correct_stt(raw_text, stage=stage)
    conversation_memory["size_closed_before_turn"] = bool(conversation_memory.get("size_decided"))

    print("=" * 50)
    print("CUSTOMER SAID (raw):", raw_text)
    print("CUSTOMER SAID (corrected):", user_text)
    print("MEMORY:", conversation_memory)

    if stage == "greeting":
        if asks_to_call_later(user_text):
            return advisor_followup_twiml()
        if is_positive(user_text) or available_to_talk(user_text):
            conversation_memory["stage"] = "qualify"
            twiml = f"""
<Response>

{gather_block(
    "Thanks for taking the call. " + CONFIGURATION_QUESTION
)}

</Response>
"""
            return Response(content=twiml, media_type="application/xml")

        if is_clear_negative(user_text):
            conversation_memory["stage"] = "closed"
            return Response(
                content=polite_exit_twiml(
                    "Alright, thanks for letting me know. Have a lovely day."
                ),
                media_type="application/xml",
            )

        return Response(
            content=qualification_twiml("Is now a good time for a quick call?"),
            media_type="application/xml",
        )

    if stage == "qualify":
        stopped = polite_stop(user_text)
        if stopped:
            return stopped
        if needs_time(user_text):
            return time_to_decide_twiml()
        floor = response_for(reply_for_floor_plan(user_text))
        if floor:
            return floor
        if wants_site_visit(user_text):
            return advisor_followup_twiml()
        if asks_about_the_visit(user_text):
            return offer_the_visit_once()
        if is_bare_configuration_word(user_text):
            return Response(
                content=qualification_twiml(CONFIGURATION_FOLLOWUP),
                media_type="application/xml",
            )
        step = route_known(known_transition(user_text))
        if step:
            return step
        note_customer_requirements(user_text)
        declined = response_for(reply_for_declined_topic(user_text))
        if declined:
            return declined
        opened = open_answer_response(user_text, from_qualify=True)
        if opened:
            return opened
        accepted = response_for(reply_for_accepted_topic(user_text))
        if accepted:
            return accepted
        purpose = response_for(reply_for_purpose(user_text))
        if purpose:
            return purpose
        if yes_to_materials(user_text):
            return share_details_twiml()
        if finished_this_topic(user_text):
            return close_or_offer_the_visit()
        question = missing_requirement_question(user_text)
        if question:
            return Response(content=qualification_twiml(question), media_type="application/xml")
        conversation_memory["stage"] = "answer_faq"
        twiml = answer_faq_and_prompt(user_text)
        return Response(content=twiml, media_type="application/xml")

    if stage == "answer_faq":
        stopped = polite_stop(user_text)
        if stopped:
            return stopped
        if needs_time(user_text):
            return time_to_decide_twiml()
        floor = response_for(reply_for_floor_plan(user_text))
        if floor:
            return floor
        if conversation_memory.get("awaiting_visit_decision"):
            conversation_memory["awaiting_visit_decision"] = False
            opened = open_answer_response(user_text)
            if opened:
                return opened
            if asks_about_the_visit(user_text):
                return offer_the_visit_once()
            if is_bare_okay(user_text):
                return visit_needs_a_clear_yes_twiml()
            if yes_to_materials(user_text):
                return share_details_twiml()
            if is_positive(user_text) or wants_site_visit(user_text):
                return advisor_followup_twiml()
            if is_clear_negative(user_text):
                return advisor_followup_twiml()
            conversation_memory["stage"] = "answer_faq"
            twiml = answer_faq_and_prompt(user_text)
            return Response(content=twiml, media_type="application/xml")

        if wants_site_visit(user_text):
            return advisor_followup_twiml()
        if asks_about_the_visit(user_text):
            return offer_the_visit_once()

        if is_bare_configuration_word(user_text):
            return Response(
                content=qualification_twiml(CONFIGURATION_FOLLOWUP),
                media_type="application/xml",
            )
        step = route_known(known_transition(user_text))
        if step:
            return step
        note_customer_requirements(user_text)
        declined = response_for(reply_for_declined_topic(user_text))
        if declined:
            return declined
        opened = open_answer_response(user_text)
        if opened:
            return opened
        accepted = response_for(reply_for_accepted_topic(user_text))
        if accepted:
            return accepted
        purpose = response_for(reply_for_purpose(user_text))
        if purpose:
            return purpose
        if yes_to_materials(user_text):
            return share_details_twiml()
        if finished_this_topic(user_text):
            return close_or_offer_the_visit()
        question = missing_requirement_question(user_text)
        if question:
            conversation_memory["stage"] = "qualify"
            return Response(content=qualification_twiml(question), media_type="application/xml")

        if is_new_faq_question(user_text):
            nudge = conversation_memory["faq_count"] >= MAX_FAQ_TURNS
            twiml = answer_faq_and_prompt(user_text, nudge_visit=nudge)
            return Response(content=twiml, media_type="application/xml")

        if declining_last_offer(user_text):
            reply = decline_followup()
            if reply is None:
                conversation_memory["stage"] = "closed"
                return Response(
                    content=polite_exit_twiml(
                        "No problem. I can ask our team to share the project details with you. "
                        "Thank you, and have a good day."
                    ),
                    media_type="application/xml",
                )
            return Response(content=qualification_twiml(reply), media_type="application/xml")

        if is_clear_negative(user_text):
            conversation_memory["stage"] = "propose_slot"
            return Response(content=propose_slot_twiml(), media_type="application/xml")

        twiml = answer_faq_and_prompt(user_text)
        return Response(content=twiml, media_type="application/xml")

    if stage == "propose_slot":
        stopped = polite_stop(user_text)
        if stopped:
            return stopped
        if needs_time(user_text):
            return time_to_decide_twiml()
        floor = response_for(reply_for_floor_plan(user_text))
        if floor:
            return floor
        declined = response_for(reply_for_declined_topic(user_text))
        if declined:
            return declined
        opened = open_answer_response(user_text)
        if opened:
            return opened
        if asks_about_the_visit(user_text):
            return offer_the_visit_once()
        if is_bare_okay(user_text):
            return visit_needs_a_clear_yes_twiml()
        if yes_to_materials(user_text):
            return share_details_twiml()
        if (is_positive(user_text) and last_question_is_visit()) or wants_site_visit(user_text):
            return advisor_followup_twiml()

        if is_new_faq_question(user_text):
            conversation_memory["stage"] = "answer_faq"
            twiml = answer_faq_and_prompt(user_text)
            return Response(content=twiml, media_type="application/xml")

        return answer_in_place(user_text)

    if stage in ("visit_day", "visit_pick_slot"):
        stopped = polite_stop(user_text)
        if stopped:
            return stopped
        if needs_time(user_text):
            return time_to_decide_twiml()
        return advisor_followup_twiml()

    conversation_memory["stage"] = "closed"
    return Response(
        content=polite_exit_twiml("Thanks for speaking with Sobha. Have a great day."),
        media_type="application/xml",
    )


# =====================================================
# ROOT
# =====================================================


@app.get("/")
async def root():
    return {
        "status": "running",
        "stage": conversation_memory["stage"],
        "faq_count": conversation_memory["faq_count"],
    }

