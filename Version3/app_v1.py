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

SARVAM_API_KEY = "KEY"
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
MODEL = "sarvam-m"
BROCHURE_FILE = "sobha-townpark-brochure.pdf"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BROCHURE_PATH = os.path.join(BASE_DIR, BROCHURE_FILE)
MAX_FAQ_TURNS = 3

# =====================================================
# MEMORY STORE
# =====================================================

conversation_memory = {
    "stage": "greeting",
    "faq_count": 0,
    "last_intent": None,
    "visit_preference": None,
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
        "floor", "plan", "layout", "configuration", "bhk", "bedroom",
        "sq ft", "sqft", "carpet", "super built", "unit", "apartment size",
        "1 bhk", "2 bhk", "3 bhk", "4 bhk",
    ],
}

INTENT_QUERY_EXPANSION = {
    "amenities": "clubhouse swimming pool sports gym kids play area wellness amenities facilities",
    "pricing": "price cost rate pricing payment offer per square feet",
    "location": "location Electronic City Hosur Road Bengaluru connectivity nearby schools offices",
    "floorplan": "floor plan layout 1 BHK 2 BHK 3 BHK configuration carpet area",
    "general": "SOBHA Townpark luxury residential apartment New York themed",
}

FALLBACK_RESPONSES = {
    "amenities": (
        "SOBHA Townpark offers premium amenities such as a clubhouse, swimming pool, "
        "sports courts, landscaped gardens and dedicated kids play areas."
    ),
    "pricing": (
        "Pricing depends on the apartment size and configuration; our sales team can "
        "share the latest rates and special offers."
    ),
    "location": (
        "SOBHA Townpark is near Electronic City on Hosur Road in Bengaluru, with strong "
        "connectivity to major IT hubs and schools."
    ),
    "floorplan": (
        "The project includes thoughtfully designed 2 BHK and 3 BHK layouts with modern "
        "interiors and spacious living areas."
    ),
    "general": (
        "SOBHA Townpark is a luxury residential destination near Electronic City offering "
        "premium finishes and excellent amenities."
    ),
}

CANONICAL_NO_INFO = (
    "I'm sorry, I don't have that detail right now; our sales team can share the latest information."
)

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
    """Retrieve the best matching brochure chunks using hybrid scoring."""
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
    print("TOP SCORE:", round(scored[0][1], 4) if scored else 0)
    print("CONTEXT PREVIEW:", context[:400])
    print("=" * 50)
    return context[:2200]


THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"


def strip_reasoning_blocks(text):
    """Remove Sarvam chain-of-thought blocks and return any spoken answer."""
    if not text:
        return ""

    cleaned = text.strip()

    # If the model put the answer AFTER a closing think tag, keep only that part.
    split_pattern = "(?i)" + re.escape(THINK_CLOSE) + "|" + re.escape("</think>")
    parts = re.split(split_pattern, cleaned)
    if len(parts) > 1:
        tail = parts[-1].strip()
        if tail:
            cleaned = tail

    # Remove full thinking blocks (open + close).
    thinking_patterns = [
        r"(?is)<think>.*?</think>",
        "(?is)" + re.escape(THINK_OPEN) + r".*?" + re.escape(THINK_CLOSE),
        r"(?is)<thinking>.*?</thinking>",
        r"(?is)<reasoning>.*?</reasoning>",
    ]
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, " ", cleaned)

    # Remove unclosed thinking blocks (happens when max_tokens cuts off mid-think).
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
        "the user asked",
        "first, looking at",
        "first looking at",
        "i need to go through",
        "i need to check",
        "looking at the",
        "let's tackle",
        "lets tackle",
        "the brochure mentions",
        "the question is",
    ]
    if any(text_lower.startswith(prefix) for prefix in reasoning_starts):
        return True

    reasoning_markers = [
        "the user is asking",
        "brochure details",
        "provided brochure",
        "i need to go through",
        "let's tackle this",
        "lets tackle this",
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
        return CANONICAL_NO_INFO

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
    """Return a safe fallback answer when brochure retrieval is insufficient."""
    return FALLBACK_RESPONSES.get(intent, CANONICAL_NO_INFO)


def generate_response(user_query, context, intent):
    """Use the LLM to answer from retrieved brochure context only."""
    if not context or len(context.strip()) < 40:
        return fallback_response(intent)

    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are a courteous real estate sales executive calling on behalf of SOBHA Townpark. "
        "Speak naturally like a human and less robotic. "
        "Answer the customer's question using ONLY the brochure details provided below. "
        "Reply with ONLY the final spoken answer. Do NOT use thinking tags, reasoning, analysis, or bullet points. "
        "Speak naturally in one or two short sentences suitable for a phone call. "
        "Do not mention brochures, documents, context, or AI. "
        "Do not guess or invent numbers, prices, sizes, or amenities that are not explicitly stated. "
        f"If the brochure details do not contain the answer, reply exactly: {CANONICAL_NO_INFO}"
    )

    user_prompt = (
        f"Customer question: {user_query}\n"
        f"Topic: {intent}\n\n"
        f"Brochure details:\n{context}\n\n"
        "Reply now with only the sentence(s) you will speak to the customer on the phone."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
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
        if not cleaned_answer:
            print("REASONING-ONLY OR INVALID LLM OUTPUT -> USING FALLBACK")
            return fallback_response(intent)
        return cleaned_answer
    except Exception as exc:
        print("SARVAM ERROR:", str(exc))
        return fallback_response(intent)


def is_negative(text):
    """Detect a clear negative or refusal."""
    text_lower = text.lower().strip()
    negative_clues = [
        "no", "nope", "nah", "not now", "later", "busy", "don't", "dont",
        "cannot", "can't", "nothing", "that's all", "thats all", "no more",
        "no thanks", "not interested",
    ]
    return any(
        text_lower == clue or text_lower.startswith(f"{clue} ")
        for clue in negative_clues
    )


def is_positive(text):
    """Detect whether the customer gave a positive response."""
    text_lower = text.lower()
    negative_clues = [
        "no", "not", "later", "busy", "don't", "dont", "cannot", "can't",
        "nope", "nah", "nothing", "that's all", "thats all", "no more",
    ]
    if any(negative in text_lower for negative in negative_clues):
        return False
    return any(
        word in text_lower
        for word in ["yes", "yeah", "sure", "ok", "okay", "fine", "great", "please", "yep", "yup"]
    )


def wants_site_visit(text):
    """Detect whether the customer is ready to schedule a visit."""
    text_lower = text.lower()
    visit_clues = [
        "visit", "site visit", "schedule", "book", "appointment", "come",
        "see", "tour", "slot", "meet",
    ]
    if any(clue in text_lower for clue in visit_clues):
        return True
    return is_positive(text)


def is_done_with_questions(text):
    """Detect whether the customer has no more FAQ questions."""
    text_lower = text.lower()
    done_clues = [
        "no more", "nothing else", "that's all", "thats all", "no questions",
        "i'm good", "im good", "all good", "enough", "no thanks", "not now",
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


def answer_faq_and_prompt(user_text):
    """Retrieve brochure context, generate an answer, and ask a follow-up."""
    intent = detect_intent(user_text)
    conversation_memory["last_intent"] = intent
    context = retrieve_context(intent, user_text)
    answer = escape_twiml(generate_response(user_text, context, intent))
    conversation_memory["faq_count"] += 1

    follow_up = (
        "Would you like to know anything else about amenities, pricing, location or floor plans, "
        "or shall I propose a convenient site visit slot?"
    )

    return f"""
<Response>

<Say>{answer}</Say>

<Pause length="1"/>

<Gather
input="speech"
action="/handle-speech"
method="POST"
speechTimeout="auto">

<Say>{follow_up}</Say>

</Gather>

</Response>
"""

# =====================================================
# START CALL
# =====================================================


@app.post("/voice")
async def voice():
    """Start the call and begin the greeting stage."""
    reset_memory()

    twiml = """
<Response>

<Gather
input="speech"
action="/handle-speech"
method="POST"
speechTimeout="auto">

<Say>
Hi, I'm calling from Sobha Limited. We're reaching out about SOBHA Townpark near Electronic City on Hosur Road in Bengaluru, a New York-themed luxury apartment community. Is this a good time to speak?
</Say>

</Gather>

</Response>
"""
    return Response(content=twiml, media_type="application/xml")


# =====================================================
# HANDLE SPEECH
# =====================================================


@app.post("/handle-speech")
async def handle_speech(SpeechResult: str = Form(...)):
    """Advance the call through greeting, qualify, FAQ, slot proposal, confirmation, and close."""
    user_text = SpeechResult.strip()
    print("=" * 50)
    print("CUSTOMER SAID:")
    print(user_text)
    print("MEMORY:")
    print(conversation_memory)

    stage = conversation_memory.get("stage", "greeting")

    if stage == "greeting":
        if is_positive(user_text):
            conversation_memory["stage"] = "qualify"
            twiml = """
<Response>

<Gather
input="speech"
action="/handle-speech"
method="POST"
speechTimeout="auto">

<Say>
Great. Would you like to know about amenities, pricing or location at SOBHA Townpark?
</Say>

</Gather>

</Response>
"""
            return Response(content=twiml, media_type="application/xml")

        conversation_memory["stage"] = "closed"
        twiml = """
<Response>

<Say>
Thank you for your time. Have a great day.
</Say>

<Hangup/>

</Response>
"""
        return Response(content=twiml, media_type="application/xml")

    if stage == "qualify":
        conversation_memory["stage"] = "answer_faq"
        twiml = answer_faq_and_prompt(user_text)
        return Response(content=twiml, media_type="application/xml")

    if stage == "answer_faq":
        if wants_site_visit(user_text) or is_done_with_questions(user_text) or is_negative(user_text):
            conversation_memory["stage"] = "propose_slot"
            twiml = """
<Response>

<Gather
input="speech"
action="/handle-speech"
method="POST"
speechTimeout="auto">

<Say>
Would you like me to propose a convenient site visit slot for SOBHA Townpark?
</Say>

</Gather>

</Response>
"""
            return Response(content=twiml, media_type="application/xml")

        if conversation_memory["faq_count"] >= MAX_FAQ_TURNS:
            conversation_memory["stage"] = "propose_slot"
            twiml = """
<Response>

<Gather
input="speech"
action="/handle-speech"
method="POST"
speechTimeout="auto">

<Say>
Thank you. Would you like me to propose a convenient site visit slot for SOBHA Townpark?
</Say>

</Gather>

</Response>
"""
            return Response(content=twiml, media_type="application/xml")

        twiml = answer_faq_and_prompt(user_text)
        return Response(content=twiml, media_type="application/xml")

    if stage == "propose_slot":
        if is_positive(user_text):
            conversation_memory["stage"] = "confirm_slot"
            twiml = """
<Response>

<Gather
input="speech"
action="/handle-speech"
method="POST"
speechTimeout="auto">

<Say>
Great. Which day and time would suit you best for a visit? Morning, afternoon or evening?
</Say>

</Gather>

</Response>
"""
            return Response(content=twiml, media_type="application/xml")

        conversation_memory["stage"] = "closed"
        twiml = """
<Response>

<Say>
No problem. I will have our sales advisor share the latest details with you shortly. Thank you for your time.
</Say>

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

<Say>
Excellent. I have noted {safe_preference}. Our sales advisor will contact you shortly to confirm the visit.
</Say>

<Hangup/>

</Response>
"""
        return Response(content=twiml, media_type="application/xml")

    twiml = """
<Response>

<Say>
Thank you for speaking with Sobha. Have a great day.
</Say>

<Hangup/>

</Response>
"""
    return Response(content=twiml, media_type="application/xml")


# =====================================================
# ROOT
# =====================================================


@app.get("/")
async def root():
    return {"status": "running", "stage": conversation_memory["stage"]}
