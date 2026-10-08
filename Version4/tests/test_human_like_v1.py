"""Offline human-like conversation checks. No Twilio and no Sarvam network calls."""

import asyncio
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app_v3  # noqa: E402

OLD_SUFFIX = "anything else you'd like to know about the project"
AMENITIES_DUMP = "clubhouses, swimming pools"
REPORT_PATH = Path(__file__).with_name("human_likeness_v1_report.md")

REPLIES = {
    "I'm looking for a 2BHK.": (
        "You want a 2 BHK at SOBHA Townpark, near Electronic City on Hosur Road. "
        "Is this mainly for your own use?"
    ),
    "I'm looking around South Bengaluru.": (
        "You mentioned South Bengaluru. "
        "SOBHA Townpark is near Electronic City on Hosur Road, "
        "and I don't have a confirmed distance for that comparison. "
        "Is South Bengaluru your main priority?"
    ),
    "How far is Townpark from Whitefield?": (
        "I don't have a confirmed distance from Whitefield. "
        "SOBHA Townpark is located Near Electronic City on Hosur Road, Bengaluru. "
        "Is proximity to Whitefield your main priority?"
    ),
    "I'm looking for a 2BHK in South Bengaluru.": (
        "You want a 2 BHK in South Bengaluru. "
        "SOBHA Townpark is located Near Electronic City on Hosur Road, Bengaluru, "
        "and it includes 2 BHK apartments. Would you like the starting price?"
    ),
    "I'm working in Whitefield and I don't want something too far.": (
        "I understand. You don't want something too far from Whitefield. "
        "I don't have a confirmed distance from Whitefield. Is location your main priority?"
    ),
    "How far is the project from Whitefield?": (
        "I don't have a confirmed distance from Whitefield. "
        "SOBHA Townpark is located Near Electronic City on Hosur Road, Bengaluru. "
        "Is location your main priority?"
    ),
    "What is the price?": (
        "SOBHA Townpark starts at approximately INR 1.8 Crore onwards. "
        "Is that within what you want to spend?"
    ),
    "Is there any offer?": "I don't have a confirmed offer in the details I have.",
    "Sobha seems expensive.": (
        "I understand. Price is the concern. "
        "SOBHA Townpark starts at approximately INR 1.8 Crore onwards. "
        "What budget range should I keep in mind?"
    ),
    "I'm looking for an investment.": (
        "I understand this is for investment. "
        "SOBHA Townpark starts at approximately INR 1.8 Crore onwards. "
        "Is investment the main purpose?"
    ),
    "I want something for my own use.": (
        "I understand this is for your own use. "
        "SOBHA Townpark is located Near Electronic City on Hosur Road, Bengaluru. "
        "Which configuration matters most for you?"
    ),
    "What is the exact possession date?": (
        "I don't have a confirmed possession date in the details I have."
    ),
    "I want a 3 BHK instead.": (
        "Understood, you now want a 3 BHK. "
        "SOBHA Townpark has 3 BHK apartments. Should I stay with the 3 BHK?"
    ),
    "Yes.": (
        "The 3 BHK Luxe is 1506 to 1514 sq. ft., and the Grande is 1842 to 2547 sq. ft. "
        "Would you like the starting price?"
    ),
    "Configuration.": (
        "For the 3 BHK, Luxe is 1506 to 1514 sq. ft. and Grande is 1842 to 2547 sq. ft. "
        "Which of those two should I stay with?"
    ),
    "One minute.": "Sure, I will wait. Is now still a good time?",
    "Just one minute.": "Sure, I will wait.",
    "Can I ask something?": "Yes, please go ahead.",
}

CALLS = []


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        user = self.payload["messages"][1]["content"]
        said = user.split("Customer said: ", 1)[1].split("\n", 1)[0]
        text = REPLIES.get(said) or (
            "The project offers clubhouses, swimming pools and a generic summary."
        )
        return {
            "choices": [
                {
                    "message": {"content": text},
                    "finish_reason": "stop",
                }
            ]
        }


def _fake_post(url, headers=None, json=None, timeout=None):
    CALLS.append({"url": url, "json": json})
    return _Response(json)


class _Request:
    def __init__(self, speech):
        self._form = {"SpeechResult": speech, "UnstableSpeechResult": ""}

    async def form(self):
        return self._form


def _spoken(xml):
    parts = re.findall(r"<Say[^>]*>(.*?)</Say>", xml, flags=re.S)
    text = " ".join(parts)
    return (
        text.replace("&apos;", "'")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def _turn(speech):
    response = asyncio.run(
        app_v3.handle_speech(_Request(speech), SpeechResult=speech)
    )
    xml = response.body.decode() if isinstance(response.body, bytes) else response.body
    return _spoken(xml), xml, app_v3.conversation_memory["stage"]


def _score(case, spoken, stage):
    low = spoken.lower()
    questions = spoken.count("?")
    point_hit = all(part.lower() in low for part in case["points"])
    if case.get("ack_optional"):
        acknowledgement = 2
    else:
        acknowledgement = 2 if any(word in low for word in case.get("ack", ("understand", "understood"))) else 0
    matched = 2 if point_hit else 0
    follow = 2 if questions <= case.get("max_questions", 1) else 0
    no_dump = 0 if AMENITIES_DUMP in low else 2
    no_repeat = 2 if OLD_SUFFIX not in low and questions <= 1 else 0
    banned = ("discount", "cashback", "free parking", "free registration", "appreciation")
    grounded = 0 if any(term in low for term in banned) else 2
    safe_stage = 2 if stage == case["stage"] else 0
    scores = {
        "A": acknowledgement,
        "B": matched,
        "C": follow,
        "D": no_dump,
        "E": no_repeat,
        "F": grounded,
        "G": safe_stage,
    }
    scores["total"] = sum(scores.values())
    return scores


CASES = []


@pytest.fixture(scope="session", autouse=True)
def _write_report():
    yield
    if not CASES:
        return
    lines = [
        "# Human-Likeness V1 regression",
        "",
        "Offline only. Each column is scored 0–2. Maximum is 14.",
        "",
        "| Case | A | B | C | D | E | F | G | Total |",
        "|------|---|---|---|---|---|---|---|-------|",
    ]
    for speech, _spoken, _stage, scores in CASES:
        lines.append(
            "| {speech} | {A} | {B} | {C} | {D} | {E} | {F} | {G} | {total} |".format(
                speech=speech.replace("|", "/"),
                **scores,
            )
        )
    lines.extend(
        [
            "",
            "A acknowledgement, B matches the customer's point, C at most one follow-up,",
            "D no project dump, E no repeated suffix, F grounded, G stage safety.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    CALLS.clear()
    app_v3.reset_memory()
    monkeypatch.setattr(app_v3.requests, "post", _fake_post)
    yield


def _record(name, speech, **prepare):
    if prepare:
        app_v3.conversation_memory.update(prepare)
    spoken, xml, stage = _turn(speech)
    return spoken, xml, stage


def test_model_stt_and_tts_unchanged():
    source = (ROOT / "app_v3.py").read_text(encoding="utf-8")
    assert app_v3.MODEL == "sarvam-105b-conversations"
    assert 'speechModel="experimental_utterances"' in source
    assert 'language="en-IN"' in source
    assert 'os.getenv("TWILIO_SAY_VOICE", "Polly.Aditi")' in source
    assert "BOOKABLE_SLOTS" in source


def test_prompt_requires_one_question_and_no_invention():
    app_v3.conversation_memory["stage"] = "answer_faq"
    _turn("What is the price?")
    system = CALLS[-1]["json"]["messages"][0]["content"]
    user = CALLS[-1]["json"]["messages"][1]["content"]
    assert "at most one" in system
    assert "Do not invent" in system
    assert "no confirmed offer" in system.lower()
    assert "What is the price?" in user
    assert CALLS[-1]["json"]["model"] == "sarvam-105b-conversations"


def test_weak_retrieval_still_uses_the_model(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(app_v3.requests, "post", explode)
    spoken = app_v3.generate_response("Is there any offer?", "pricing", "", 0.01)
    assert "confirmed offer" in spoken.lower()
    assert AMENITIES_DUMP not in spoken.lower()


def test_failed_distance_lookup_does_not_dump_amenities(monkeypatch):
    monkeypatch.setattr(app_v3.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    spoken = app_v3.generate_response(
        "How far is the project from Whitefield?",
        "location",
        "",
        0.01,
    )
    assert "whitefield" in spoken.lower() or "distance" in spoken.lower()
    assert "confirmed distance" in spoken.lower()
    assert AMENITIES_DUMP not in spoken.lower()


@pytest.mark.parametrize(
    ("speech", "stage", "points", "expect_stage", "ack_optional", "max_questions", "prepare"),
    [
        (
            "I'm looking for a 2BHK.",
            "qualify",
            ["2 bhk", "townpark"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "I'm looking around South Bengaluru.",
            "qualify",
            ["south bengaluru"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "I'm looking for a 2BHK in South Bengaluru.",
            "qualify",
            ["2 bhk", "south bengaluru"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "I'm working in Whitefield and I don't want something too far.",
            "answer_faq",
            ["whitefield", "distance"],
            "answer_faq",
            False,
            1,
            {},
        ),
        (
            "How far is the project from Whitefield?",
            "answer_faq",
            ["whitefield", "confirmed distance"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "What is the price?",
            "answer_faq",
            ["1.8 crore"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "Is there any offer?",
            "answer_faq",
            ["confirmed offer"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "Sobha seems expensive.",
            "answer_faq",
            ["price", "1.8 crore"],
            "answer_faq",
            False,
            1,
            {},
        ),
        (
            "I need to check with my family.",
            "answer_faq",
            ["take your time", "details"],
            "closed",
            True,
            0,
            {},
        ),
        (
            "I'll think about it.",
            "answer_faq",
            ["take your time", "details"],
            "closed",
            True,
            0,
            {},
        ),
        (
            "I'm looking for an investment.",
            "answer_faq",
            ["investment", "1.8 crore"],
            "answer_faq",
            False,
            1,
            {},
        ),
        (
            "I want something for my own use.",
            "answer_faq",
            ["own use"],
            "answer_faq",
            False,
            1,
            {},
        ),
        (
            "What is the exact possession date?",
            "answer_faq",
            ["confirmed possession"],
            "answer_faq",
            True,
            0,
            {},
        ),
        (
            "I'm not sure what I want.",
            "qualify",
            ["one, two, three, and four bhk"],
            "qualify",
            True,
            1,
            {},
        ),
        (
            "I want a 3 BHK instead.",
            "answer_faq",
            ["3 bhk"],
            "answer_faq",
            False,
            1,
            {"customer_locality": "South Bengaluru", "customer_configuration": "2 BHK"},
        ),
        (
            "I want to schedule a site visit.",
            "answer_faq",
            ["which day"],
            "visit_day",
            True,
            1,
            {},
        ),
        (
            "I'm working in Whitefield and I don't want something too far.",
            "qualify",
            ["whitefield", "distance"],
            "answer_faq",
            False,
            1,
            {},
        ),
        (
            "What is the price?",
            "qualify",
            ["1.8 crore"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "Is there any offer?",
            "qualify",
            ["confirmed offer"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "How far is Townpark from Whitefield?",
            "qualify",
            ["whitefield", "confirmed distance"],
            "answer_faq",
            True,
            1,
            {},
        ),
        (
            "I'm looking for an investment.",
            "qualify",
            ["investment", "1.8 crore"],
            "answer_faq",
            False,
            1,
            {},
        ),
        (
            "I want to schedule a site visit.",
            "qualify",
            ["which day"],
            "visit_day",
            True,
            1,
            {},
        ),
        (
            "I'll think about it.",
            "qualify",
            ["take your time", "details"],
            "closed",
            True,
            0,
            {},
        ),
    ],
)
def test_customer_turn(speech, stage, points, expect_stage, ack_optional, max_questions, prepare):
    app_v3.conversation_memory["stage"] = stage
    spoken, xml, got_stage = _record(speech, speech, **prepare)
    case = {
        "points": points,
        "stage": expect_stage,
        "ack_optional": ack_optional,
        "max_questions": max_questions,
        "ack": ("understand", "understood"),
    }
    scores = _score(case, spoken, got_stage)
    CASES.append((speech, spoken, got_stage, scores))
    assert OLD_SUFFIX not in spoken.lower()
    assert AMENITIES_DUMP not in spoken.lower()
    assert "which area in or around bengaluru" not in spoken.lower()
    if speech == "I'm looking for a 2BHK.":
        assert "two or three bhk" not in spoken.lower()
    if speech == "I'm looking around South Bengaluru.":
        assert "what configuration" not in spoken.lower()
        assert "two or three bhk" not in spoken.lower()
    assert scores["total"] == 14, (speech, spoken, scores)
    if speech.startswith("I want to schedule"):
        assert app_v3.conversation_memory["selected_slot"] is None
        assert "booked" not in spoken.lower()
        assert CALLS == []
    if speech == "I want a 3 BHK instead.":
        assert app_v3.conversation_memory["customer_configuration"] == "3 BHK"
        assert "3 BHK" in CALLS[-1]["json"]["messages"][1]["content"]


def test_greeting_asks_configuration_not_area():
    app_v3.conversation_memory["stage"] = "greeting"
    spoken, _, stage = _turn("Yes, this is a good time.")
    assert stage == "qualify"
    assert "townpark" in spoken.lower()
    assert "one, two, three, and four bhk" in spoken.lower()
    assert "which area" not in spoken.lower()
    assert "amenities, pricing" not in spoken.lower()
    assert CALLS == []


def test_greeting_accepts_available_without_yes():
    app_v3.conversation_memory["stage"] = "greeting"
    spoken, _, stage = _turn("I can speak now.")
    assert stage == "qualify"
    assert "one, two, three, and four bhk" in spoken.lower()
    assert "lovely day" not in spoken.lower()
    assert CALLS == []


def test_greeting_still_closes_on_not_now():
    app_v3.conversation_memory["stage"] = "greeting"
    spoken, _, stage = _turn("Not now.")
    assert stage == "closed"
    assert "lovely day" in spoken.lower()


def test_greeting_answers_a_hold_and_stays_open():
    app_v3.conversation_memory["stage"] = "greeting"
    app_v3.conversation_memory["last_agent_question"] = "Is now a good time for a quick call?"
    spoken, _, stage = _turn("One minute.")
    assert stage == "greeting"
    assert CALLS
    assert "lovely day" not in spoken.lower()
    assert "sure, i will wait" in spoken.lower()
    system = CALLS[-1]["json"]["messages"][0]["content"]
    user = CALLS[-1]["json"]["messages"][1]["content"]
    assert "say you will wait" in system.lower()
    assert "Current stage: greeting" in user


def test_qualification_answers_a_hold_or_a_question():
    for speech in ("Just one minute.", "Can I ask something?"):
        app_v3.reset_memory()
        CALLS.clear()
        app_v3.conversation_memory.update({
            "stage": "qualify",
            "configuration_asked": True,
            "last_agent_question": "Which configuration are you looking for?",
        })
        spoken, _, stage = _turn(speech)
        assert stage == "answer_faq"
        assert CALLS
        assert "which configuration are you looking for" not in spoken.lower()


def test_second_okay_narrows_the_bhk_question():
    app_v3.conversation_memory["stage"] = "qualify"
    app_v3.conversation_memory["configuration_asked"] = True
    app_v3.conversation_memory["last_agent_question"] = (
        "Are you mainly looking for a two or three BHK?"
    )
    spoken, _, stage = _turn("Okay.")
    assert stage == "qualify"
    assert spoken.strip() == "One, two, three, or four BHK?"
    assert "this is regarding" not in spoken.lower()
    assert CALLS == []


def test_yes_continues_the_last_question():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "3 BHK",
        "last_intent": "floorplan",
        "last_agent_question": "Would you prefer the Luxe or Grande configuration?",
    })
    _turn("Yes.")
    user = CALLS[-1]["json"]["messages"][1]["content"]
    system = CALLS[-1]["json"]["messages"][0]["content"]
    assert "Would you prefer the Luxe or Grande configuration?" in user
    assert "Customer said: Yes." in user
    assert "Detected topic: floorplan" in user
    assert "Use only these sizes for the stated configuration" in user
    assert "3 BHK Luxe is 1506 to 1514" in user
    assert "do not ask that question again" in system.lower()


def test_bhk_choice_is_kept_and_not_asked_again():
    app_v3.conversation_memory["stage"] = "qualify"
    app_v3.conversation_memory["configuration_asked"] = True
    app_v3.conversation_memory["last_agent_question"] = (
        "Are you mainly looking for a two or three BHK?"
    )
    spoken, _, stage = _turn("3 BHK.")
    assert app_v3.correct_stt("3 BHK.", stage="qualify") == "3 BHK."
    assert app_v3.conversation_memory["customer_configuration"] == "3 BHK"
    assert stage == "answer_faq"
    assert "two or three" not in spoken.lower()
    assert "Customer said: 3 BHK." in CALLS[-1]["json"]["messages"][1]["content"]


def test_size_answer_is_not_rewritten_as_pricing():
    app_v3.conversation_memory["last_agent_question"] = (
        "Which of these size ranges interests you more?"
    )
    assert app_v3.correct_stt("1842 square feet.", stage="answer_faq") == "1842 square feet."


def test_no_to_payment_does_not_start_a_visit():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "3 BHK",
        "customer_variant": "Grande",
        "last_intent": "pricing",
        "last_agent_question": "Would you like to know more about the payment options?",
    })
    spoken, _, stage = _turn("No.")
    assert stage == "answer_faq"
    assert "leave the payment plans" in spoken.lower()
    assert "grande 3 bhk" in spoken.lower()
    assert "1.8" not in spoken.lower()
    assert "site visit" not in spoken.lower()
    assert CALLS == []


def test_size_choice_stays_on_the_layout():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "3 BHK",
        "last_intent": "floorplan",
        "last_agent_question": "Which of these sizes works for you?",
    })
    _turn("Granded 1842 square feet.")
    user = CALLS[-1]["json"]["messages"][1]["content"]
    assert "Customer said: Grande 1842 square feet." in user
    assert "Detected topic: floorplan" in user
    assert "Do not mention price or payment plans." in user
    assert app_v3.conversation_memory["customer_variant"] == "Grande"


def test_configuration_reply_stays_on_the_layout_question():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "3 BHK",
        "last_intent": "floorplan",
        "last_agent_question": "Would you prefer the Luxe or Grande configuration?",
    })
    assert app_v3.correct_stt("Configuration.", stage="answer_faq") == "Configuration."
    _turn("Configuration.")
    user = CALLS[-1]["json"]["messages"][1]["content"]
    assert "Customer said: Configuration." in user
    assert "Customer said: floor plan" not in user
    assert "Detected topic: floorplan" in user
    assert "3 BHK Grande is 1842 to 2547" in user
