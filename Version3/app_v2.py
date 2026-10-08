import os
import re
import math
import fitz
import requests
from collections import Counter
from fastapi import FastAPI, Form
from fastapi.responses import Response

app = FastAPI()

# =====================================================
# CONFIG
# =====================================================

SARVAM_API_KEY = "sk_tttkw3yh_cSFe8L2vfkaLKLw1JnGgRNBT"
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
MODEL = "sarvam-m"
BROCHURE_FILE = "sobha-townpark-brochure.pdf"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BROCHURE_PATH = os.path.join(BASE_DIR, BROCHURE_FILE)
MAX_FAQ_TURNS = 8
LOW_RETRIEVAL_SCORE = 0.12

# Twilio speech hints improve STT accuracy for expected customer words
SPEECH_HINTS = (
    "amenities,amenity,pricing,price,cost,location,hosur,electronic city,"
    "configuration,configurations,floor plan,2 BHK,3 BHK,site visit,"
    "yes,no,okay,thank you,morning,afternoon,evening"
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

# =====================================================
# MEMORY STORE
# =====================================================

conversation_memory = {
    "stage": "greeting",
    "faq_count": 0,
    "last_intent": None,
    "visit_preference": None,
    "awaiting_visit_decision": False,
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
# BROCHURE PREPARATION
# =====================================================

pdf_text = load_pdf_text(BROCHURE_PATH)
brochure_chunks = chunk_text(pdf_text)
brochure_vectors = build_tfidf_vectors(brochure_chunks)

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

    for pattern, replacement in STT_MISHEAR_MAP.items():
        if re.match(pattern, normalized, flags=re.IGNORECASE):
            print(f"STT CORRECTION: '{original}' -> '{replacement}'")
            return replacement

    # After qualify prompt, short replies are usually topic picks
    if stage in ("qualify", "answer_faq") and len(normalized.split()) <= 4:
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


def gather_block(prompt_text):
    """Reusable Gather with speech hints for better STT."""
    return f"""
<Gather
input="speech"
action="/handle-speech"
method="POST"
speechTimeout="auto"
hints="{SPEECH_HINTS}"
language="en-IN">

<Say>
{prompt_text}
</Say>

</Gather>
"""


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

    if text.lower().startswith(CANONICAL_NO_INFO.lower()):
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

    return text[:320].strip()


def fallback_response(intent):
    """Return answer from project knowledge base when brochure/LLM fails."""
    return answer_from_kb(intent)


def generate_response(user_query, intent, brochure_context, top_score):
    """Answer using brochure + project KB; KB-first when brochure is weak."""
    kb_primary = should_use_kb_primary(intent, top_score, brochure_context)

    if kb_primary:
        print("USING PROJECT KB (weak or irrelevant brochure match)")
        return answer_from_kb(intent)

    kb_section = kb_context_for_intent(intent)
    combined_context = f"Project knowledge:\n{kb_section}\n\nBrochure excerpts:\n{brochure_context}"

    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are a courteous real estate sales executive calling on behalf of SOBHA Townpark. "
        "Answer using the project knowledge and brochure details provided. "
        "Prefer project knowledge for pricing, configurations and payment plans. "
        "Reply with ONLY the final spoken answer. No thinking tags, no bullet points. "
        "One or two short sentences suitable for a phone call. "
        "Speak naturally like a human and avoid being robotic."
        "Do not mention brochures, documents, context, or AI. "
        "Do not invent facts not present in the provided information."
    )

    user_prompt = (
        f"Customer question: {user_query}\n"
        f"Topic: {intent}\n\n"
        f"{combined_context}\n\n"
        "Reply now with only what you will speak to the customer."
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
    }

    try:
        response = requests.post(SARVAM_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        print("=" * 50)
        print("SARVAM RAW RESPONSE:")
        print(result)
        print("=" * 50)

        raw_answer = result["choices"][0]["message"]["content"]
        finish_reason = result["choices"][0].get("finish_reason", "")
        print("FINISH REASON:", finish_reason)
        print("RAW ANSWER PREVIEW:", raw_answer[:500])

        cleaned_answer = clean_response(raw_answer)
        if not cleaned_answer or finish_reason == "length":
            print("REASONING-ONLY / LENGTH LIMIT -> USING KB FALLBACK")
            return answer_from_kb(intent)
        return cleaned_answer
    except Exception as exc:
        print("SARVAM ERROR:", str(exc))
        return answer_from_kb(intent)


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


def is_positive(text):
    """Detect whether the customer gave a positive response."""
    text_lower = text.lower()
    if is_clear_negative(text_lower):
        return False
    if re.search(r"\b(i would like|tell me|what about|how about|know about)\b", text_lower):
        return False
    return any(
        contains_word(text_lower, word)
        for word in ["yes", "yeah", "sure", "ok", "okay", "fine", "great", "yep", "yup"]
    )


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


def answer_faq_and_prompt(user_text, nudge_visit=False):
    """Retrieve context, generate answer, and ask follow-up or visit nudge."""
    intent = detect_intent(user_text)
    conversation_memory["last_intent"] = intent
    brochure_context, top_score = retrieve_context(intent, user_text)
    answer = escape_twiml(generate_response(user_text, intent, brochure_context, top_score))
    conversation_memory["faq_count"] += 1

    if nudge_visit:
        follow_up = "Would you like me to propose a convenient site visit slot for SOBHA Townpark?"
        conversation_memory["awaiting_visit_decision"] = True
    else:
        follow_up = (
            "Would you like to know anything else about amenities, pricing, location or floor plans, "
            "or shall I propose a convenient site visit slot?"
        )

    return f"""
<Response>

<Say>{answer}</Say>

<Pause length="1"/>

{gather_block(follow_up)}

</Response>
"""


def propose_slot_twiml():
    """Ask customer if they want a site visit."""
    return f"""
<Response>

{gather_block("Would you like me to propose a convenient site visit slot for SOBHA Townpark?")}

</Response>
"""


# =====================================================
# START CALL
# =====================================================


@app.post("/voice")
async def voice():
    """Start the call and begin the greeting stage."""
    reset_memory()

    twiml = f"""
<Response>

{gather_block(
    "Hi, I'm calling from Sobha Limited. We're reaching out about SOBHA Townpark near "
    "Electronic City on Hosur Road in Bengaluru, a New York-themed luxury apartment community. "
    "Is this a good time to speak?"
)}

</Response>
"""
    return Response(content=twiml, media_type="application/xml")


# =====================================================
# HANDLE SPEECH
# =====================================================


@app.post("/handle-speech")
async def handle_speech(SpeechResult: str = Form(...)):
    """Greeting -> Qualify -> Answer FAQ -> Propose Slot -> Confirm Slot -> Closed"""
    raw_text = SpeechResult.strip()
    stage = conversation_memory.get("stage", "greeting")
    user_text = correct_stt(raw_text, stage=stage)

    print("=" * 50)
    print("CUSTOMER SAID (raw):", raw_text)
    print("CUSTOMER SAID (corrected):", user_text)
    print("MEMORY:", conversation_memory)

    if stage == "greeting":
        if is_positive(user_text):
            conversation_memory["stage"] = "qualify"
            twiml = f"""
<Response>

{gather_block(
    "Great. Would you like to know about amenities, pricing, location or floor plans at SOBHA Townpark?"
)}

</Response>
"""
            return Response(content=twiml, media_type="application/xml")

        conversation_memory["stage"] = "closed"
        twiml = """
<Response>
<Say>Thank you for your time. Have a great day.</Say>
<Hangup/>
</Response>
"""
        return Response(content=twiml, media_type="application/xml")

    if stage == "qualify":
        conversation_memory["stage"] = "answer_faq"
        twiml = answer_faq_and_prompt(user_text)
        return Response(content=twiml, media_type="application/xml")

    if stage == "answer_faq":
        if conversation_memory.get("awaiting_visit_decision"):
            conversation_memory["awaiting_visit_decision"] = False
            if is_positive(user_text) or wants_site_visit(user_text):
                conversation_memory["stage"] = "confirm_slot"
                twiml = f"""
<Response>

{gather_block(
    "Great. Which day and time would suit you best for a visit? Morning, afternoon or evening?"
)}

</Response>
"""
                return Response(content=twiml, media_type="application/xml")
            if is_clear_negative(user_text):
                conversation_memory["stage"] = "closed"
                twiml = """
<Response>
<Say>No problem. Our sales advisor will share the latest details with you shortly. Thank you for your time.</Say>
<Hangup/>
</Response>
"""
                return Response(content=twiml, media_type="application/xml")
            conversation_memory["stage"] = "answer_faq"
            twiml = answer_faq_and_prompt(user_text)
            return Response(content=twiml, media_type="application/xml")

        if wants_site_visit(user_text) or is_done_with_questions(user_text):
            conversation_memory["stage"] = "propose_slot"
            return Response(content=propose_slot_twiml(), media_type="application/xml")

        if is_new_faq_question(user_text):
            nudge = conversation_memory["faq_count"] >= MAX_FAQ_TURNS
            twiml = answer_faq_and_prompt(user_text, nudge_visit=nudge)
            if nudge:
                conversation_memory["awaiting_visit_decision"] = True
            return Response(content=twiml, media_type="application/xml")

        if is_clear_negative(user_text):
            conversation_memory["stage"] = "propose_slot"
            return Response(content=propose_slot_twiml(), media_type="application/xml")

        twiml = answer_faq_and_prompt(user_text)
        return Response(content=twiml, media_type="application/xml")

    if stage == "propose_slot":
        if is_positive(user_text) or wants_site_visit(user_text):
            conversation_memory["stage"] = "confirm_slot"
            twiml = f"""
<Response>

{gather_block(
    "Great. Which day and time would suit you best for a visit? Morning, afternoon or evening?"
)}

</Response>
"""
            return Response(content=twiml, media_type="application/xml")

        if is_new_faq_question(user_text):
            conversation_memory["stage"] = "answer_faq"
            twiml = answer_faq_and_prompt(user_text)
            return Response(content=twiml, media_type="application/xml")

        conversation_memory["stage"] = "closed"
        twiml = """
<Response>
<Say>No problem. I will have our sales advisor share the latest details with you shortly. Thank you for your time.</Say>
<Hangup/>
</Response>
"""
        return Response(content=twiml, media_type="application/xml")

    if stage == "confirm_slot":
        conversation_memory["visit_preference"] = user_text
        conversation_memory["stage"] = "closed"
        safe_preference = escape_twiml(user_text)
        twiml = f"""
<Response>
<Say>Excellent. I have noted {safe_preference}. Our sales advisor will contact you shortly to confirm the visit.</Say>
<Hangup/>
</Response>
"""
        return Response(content=twiml, media_type="application/xml")

    twiml = """
<Response>
<Say>Thank you for speaking with Sobha. Have a great day.</Say>
<Hangup/>
</Response>
"""
    return Response(content=twiml, media_type="application/xml")


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
