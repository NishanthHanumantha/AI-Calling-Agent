# =========================================================
# IMPORTS
# =========================================================

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import Response

from twilio.twiml.voice_response import VoiceResponse
from twilio.twiml.voice_response import Gather

from typing import Dict, Any

import requests
import re

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

# =========================================================
# SARVAM CONFIG
# =========================================================

SARVAM_API_KEY = "KEY"

SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"

SARVAM_MODEL = "sarvam-m"

# =========================================================
# SESSION-BASED MEMORY
# =========================================================

conversation_memory = {}

# =========================================================
# PROJECT KNOWLEDGE BASE
# =========================================================

PROJECT_KB: Dict[str, Any] = {

    "project_name": "SOBHA Townpark",

    "brand": "Sobha Limited",

    "city": "Bengaluru",

    "theme": "New York-themed luxury apartments",

    "location":
        "Near Electronic City on Hosur Road Bengaluru",

    "starting_price":
        "INR 1.8 Crore onwards",

    "configurations":
        "1 BHK, 2 BHK, 3 BHK and 4 BHK",

    "amenities":
        "clubhouses, swimming pools, sports courts, forest grove and camping grounds"
}

# =========================================================
# ALLOWED VISIT SLOTS
# =========================================================

ALLOWED_SLOTS = [

    "10:00 AM",

    "11:30 AM",

    "4:00 PM",

    "6:30 PM"
]

# =========================================================
# CLEAN AI RESPONSE
# =========================================================

def clean_ai_response(text):

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    text = text.strip()

    return text

# =========================================================
# DETECT INTENT
# =========================================================

def detect_intent(customer_text):

    text = customer_text.lower()

    if (
        "amenities" in text
        or "clubhouse" in text
        or "pool" in text
    ):

        return "amenities"

    elif (
        "price" in text
        or "pricing" in text
        or "cost" in text
    ):

        return "price"

    elif (
        "location" in text
        or "where" in text
    ):

        return "location"

    elif (
        "configuration" in text
        or "bhk" in text
    ):

        return "configurations"

    elif (
        "visit" in text
        or "site visit" in text
    ):

        return "site_visit"

    return "general"

# =========================================================
# ANSWER FROM KB
# =========================================================

def answer_from_kb(customer_text):

    intent = detect_intent(customer_text)

    if intent == "amenities":

        kb_answer = (
            "The project offers clubhouses, "
            "swimming pools, sports courts, "
            "forest grove and camping grounds."
        )

    elif intent == "price":

        kb_answer = (
            "The starting price is approximately "
            "INR 1.8 crore onwards."
        )

    elif intent == "location":

        kb_answer = (
            "The project is located near "
            "Electronic City on Hosur Road Bengaluru."
        )

    elif intent == "configurations":

        kb_answer = (
            "The project offers "
            "1 BHK, 2 BHK, 3 BHK and 4 BHK apartments."
        )

    else:

        kb_answer = (
            "Could you please clarify your question?"
        )

    # =====================================================
    # SARVAM AI CALL
    # =====================================================

    payload = {

        "model": SARVAM_MODEL,

        "messages": [

            {
                "role": "system",

                "content":
                """
                You are a professional
                Indian real estate sales assistant.

                STRICT RULES:

                1. NEVER explain reasoning.

                2. NEVER output thinking.

                3. NEVER use <think> tags.

                4. Keep answers under 25 words.

                5. ONLY generate customer-facing speech.
                """
            },

            {
                "role": "user",

                "content":
                f"""
                Customer asked:
                {customer_text}

                Correct answer:
                {kb_answer}

                Generate natural spoken reply.
                """
            }
        ],

        "temperature": 0.5
    }

    headers = {

        "Authorization":
            f"Bearer {SARVAM_API_KEY}",

        "Content-Type":
            "application/json"
    }

    try:

        response = requests.post(
            SARVAM_URL,
            headers=headers,
            json=payload
        )

        data = response.json()

        print("\n==========================")
        print("SARVAM RAW RESPONSE:")
        print(data)
        print("==========================\n")

        ai_reply = data["choices"][0]["message"]["content"]

        ai_reply = clean_ai_response(ai_reply)

        return ai_reply

    except Exception as e:

        print("\nSARVAM ERROR:")
        print(str(e))

        return kb_answer

# =========================================================
# CONVERSATION MANAGER
# =========================================================

def conversation_manager(customer_text, memory):

    customer_text_lower = customer_text.lower().strip()

    stage = memory["stage"]

    print("\n==========================")
    print("CUSTOMER SAID:")
    print(customer_text)
    print("==========================\n")

    print("\n==========================")
    print("MEMORY STATE:")
    print(memory)
    print("==========================\n")

    # =================================================
    # GREETING
    # =================================================

    if stage == "greeting":

        memory["stage"] = "interest_check"

        return """
        Hi, I'm calling from Sobha Limited.

        We're reaching out about
        SOBHA Townpark near Electronic City
        on Hosur Road in Bengaluru,
        a New York-themed luxury apartment community.

        Is this a good time to talk?
        """

    # =================================================
    # INTEREST CHECK
    # =================================================

    elif stage == "interest_check":

        if (
            "yes" in customer_text_lower
            or "yeah" in customer_text_lower
            or "okay" in customer_text_lower
            or "sure" in customer_text_lower
        ):

            memory["stage"] = "project_pitch"

            return """
            Great.

            SOBHA Townpark offers
            New York-themed luxury apartments
            near Electronic City
            with 1 to 4 BHK options
            starting at INR 1.8 Crore onwards.

            Would you like to know more about
            amenities,
            pricing,
            location,
            or site visits?
            """

        elif (
            "no" in customer_text_lower
            or "busy" in customer_text_lower
            or "later" in customer_text_lower
        ):

            memory["stage"] = "closed"

            return """
            No problem.

            Thank you for your time.

            Have a great day.
            """

        else:

            return """
            Just checking —

            is this a good time
            to speak?
            """

    # =================================================
    # PROJECT PITCH
    # =================================================

    elif stage == "project_pitch":

        memory["stage"] = "answer_faq"

        return answer_from_kb(customer_text)

    # =================================================
    # ANSWER FAQ
    # =================================================

    elif stage == "answer_faq":

        faq_reply = answer_from_kb(customer_text)

        memory["faq_count"] += 1

        if memory["faq_count"] >= 2:

            memory["stage"] = "site_visit_interest"

            return (
                f"{faq_reply} "
                "Would you like to schedule a site visit "
                "or would you like to know anything else?"
            )

        return faq_reply

    # =================================================
    # SITE VISIT INTEREST
    # =================================================

    elif stage == "site_visit_interest":

        if (
            "visit" in customer_text_lower
            or "site" in customer_text_lower
            or "yes" in customer_text_lower
        ):

            memory["stage"] = "visit_day"

            return """
            Which day would you prefer
            for the visit?
            """

        elif (
            "anything else" in customer_text_lower
            or "know more" in customer_text_lower
        ):

            memory["stage"] = "answer_faq"

            return """
            Sure.

            Please let me know
            what you would like to know.
            """

        else:

            memory["stage"] = "answer_faq"

            return answer_from_kb(customer_text)

    # =================================================
    # VISIT DAY
    # =================================================

    elif stage == "visit_day":

        memory["visit_day"] = customer_text

        memory["stage"] = "visit_slot"

        slots_text = ", ".join(ALLOWED_SLOTS)

        return f"""
        Which slot would you prefer
        for the visit?

        Please choose from the available slots:

        {slots_text}
        """

    # =================================================
    # VISIT SLOT
    # =================================================

    elif stage == "visit_slot":

        memory["visit_slot"] = customer_text

        memory["stage"] = "confirmation"

        return f"""
        Your visit has been scheduled for
        {memory['visit_day']}
        at
        {memory['visit_slot']}.

        Our sales representative
        will contact you shortly
        and provide you the details.
        """

    # =================================================
    # CONFIRMATION
    # =================================================

    elif stage == "confirmation":

        memory["stage"] = "closed"

        return """
        Thank you for your time.

        Have a good day.
        """

    # =================================================
    # CLOSED
    # =================================================

    elif stage == "closed":

        return """
        Thank you.

        Goodbye.
        """

    # =================================================
    # FALLBACK
    # =================================================

    return """
    Sorry,
    could you please repeat that?
    """

# =========================================================
# VOICE ROUTE
# =========================================================

@app.post("/voice")
async def voice(request: Request):

    form = await request.form()

    call_sid = form.get("CallSid")

    # =====================================================
    # CREATE NEW MEMORY FOR EACH CALL
    # =====================================================

    conversation_memory[call_sid] = {

        "stage": "greeting",

        "faq_count": 0,

        "visit_day": None,

        "visit_slot": None
    }

    memory = conversation_memory[call_sid]

    ai_reply = conversation_manager("", memory)

    response = VoiceResponse()

    gather = Gather(
        input="speech",
        action="/process",
        method="POST",
        speech_timeout="auto"
    )

    gather.say(
        ai_reply,
        voice="Polly.Aditi",
        language="en-IN"
    )

    response.append(gather)

    return Response(
        content=str(response),
        media_type="application/xml"
    )

# =========================================================
# PROCESS ROUTE
# =========================================================

@app.post("/process")
async def process(request: Request):

    form = await request.form()

    call_sid = form.get("CallSid")

    customer_text = form.get(
        "SpeechResult",
        ""
    )

    # =====================================================
    # GET CALL MEMORY
    # =====================================================

    memory = conversation_memory.get(call_sid)

    if memory is None:

        conversation_memory[call_sid] = {

            "stage": "greeting",

            "faq_count": 0,

            "visit_day": None,

            "visit_slot": None
        }

        memory = conversation_memory[call_sid]

    ai_reply = conversation_manager(
        customer_text,
        memory
    )

    response = VoiceResponse()

    # =====================================================
    # END CALL IF CLOSED
    # =====================================================

    if memory["stage"] == "closed":

        response.say(
            ai_reply,
            voice="Polly.Aditi",
            language="en-IN"
        )

        response.hangup()

    else:

        gather = Gather(
            input="speech",
            action="/process",
            method="POST",
            speech_timeout="auto"
        )

        gather.say(
            ai_reply,
            voice="Polly.Aditi",
            language="en-IN"
        )

        response.append(gather)

    return Response(
        content=str(response),
        media_type="application/xml"
    )