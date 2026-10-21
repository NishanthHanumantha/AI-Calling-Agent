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
    "I'm okay with anything.": (
        "Since you are flexible, the 2 BHK Grande range gives more space. "
        "Would you prefer the larger end of that range?"
    ),
    "Yes, that would be fine.": "Would the 1678 sq. ft. option suit your needs?",
    "Not sure about it.": (
        "The 2 BHK Grande range offers two well planned sizes. "
        "Would you prefer the larger or the more compact option?"
    ),
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
            ["sales advisor", "have a good day"],
            "closed",
            True,
            0,
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
            ["sales advisor", "have a good day"],
            "closed",
            True,
            0,
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
        "ack": ("understand", "understood", "got it"),
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
    assert CALLS == []
    assert "good time" in spoken.lower()
    assert "sq" not in spoken.lower()
    assert "lovely day" not in spoken.lower()


def test_busy_or_call_later_closes_with_the_advisor():
    for speech in ("No, I am busy. Can you call later?", "No, I am busy."):
        app_v3.reset_memory()
        CALLS.clear()
        app_v3.conversation_memory["stage"] = "greeting"
        spoken, _, stage = _turn(speech)
        assert stage == "closed", speech
        assert "sales advisor" in spoken.lower()
        assert "have a good day" in spoken.lower()
        assert "good time" not in spoken.lower()
        assert CALLS == []

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory["stage"] = "qualify"
    spoken, _, stage = _turn("Can you call me later?")
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "have a good day" in spoken.lower()
    assert CALLS == []


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


def test_flexible_yes_closes_the_size_and_the_next_reply_leaves_it():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "2 BHK",
        "last_intent": "floorplan",
        "last_agent_question": "Which of these sizes suits you better?",
        "last_spoken_reply": (
            "We have 2 BHK options in Luxe and Grande. "
            "The Luxe is 1240 to 1248 sq. ft. and the Grande is 1337 to 1678 sq. ft. "
            "Which of these sizes suits you better?"
        ),
    })
    spoken, _, stage = _turn("I'm okay with anything.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["customer_variant"] == "Grande"
    assert app_v3.conversation_memory["size_decided"] is True
    assert app_v3.conversation_memory["size_closed_before_turn"] is True
    low = spoken.lower()
    assert "keep the grande 2 bhk range" in low
    assert "live in" in low
    assert "larger" not in low
    assert "smaller" not in low
    assert "what else would you like to know" not in low

    spoken, _, stage = _turn("Yes, that would be fine.")
    user = CALLS[-1]["json"]["messages"][1]["content"]
    assert stage == "answer_faq"
    assert "Size already chosen: Grande 2 BHK" in user
    assert "one range, not two separate sizes" in user
    assert "do not ask about size" in user.lower()
    low = spoken.lower()
    assert "1678" not in low
    assert "larger" not in low
    assert "layout" not in low
    assert "sq" not in low
    assert "location" in low
    assert "what else would you like to know" not in low

    spoken, _, _ = _turn("Not sure about it.")
    low = spoken.lower()
    assert app_v3.conversation_memory["size_decided"] is True
    assert "two well planned" not in low
    assert "larger" not in low
    assert "sq" not in low
    assert "what else would you like to know" not in low
    assert "live in" in low


def test_fine_with_both_ranges_leaves_the_size_open():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "2 BHK",
        "last_agent_question": "Which of these two ranges interests you more?",
        "last_spoken_reply": (
            "For 2 BHK, we have Luxe at 1240 to 1248 sq. ft. and Grande at 1337 to 1678 sq. ft. "
            "Which of these two ranges interests you more?"
        ),
    })
    spoken, _, stage = _turn("I'm fine with both.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["size_decided"] is False
    assert app_v3.conversation_memory["customer_variant"] is None
    low = spoken.lower()
    assert "leave the size open" in low
    assert "grande" not in low
    assert "live in" in low


def test_for_living_is_the_purpose_line():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "2 BHK",
        "last_agent_question": "Are you looking at this to live in, or as an investment?",
    })
    spoken, _, stage = _turn("It's for living.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["customer_purpose"] == "live"
    low = spoken.lower()
    assert "own use" in low
    assert "1.8 crore" in low
    assert "grande" not in low
    assert "?" not in spoken


def test_fine_with_both_confirms_the_four_bhk_range_on_the_same_turn():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "4 BHK",
        "last_intent": "floorplan",
        "last_agent_question": "Would you prefer a larger or smaller space within this?",
        "last_spoken_reply": (
            "The 4 BHK Grande spans 2203 to 2846 sq. ft. as one range. "
            "Would you prefer a larger or smaller space within this?"
        ),
    })
    spoken, _, stage = _turn("I am fine with both.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["customer_variant"] == "Grande"
    assert app_v3.conversation_memory["size_decided"] is True
    assert app_v3.conversation_memory["size_closed_before_turn"] is True
    low = spoken.lower()
    assert "keep the grande 4 bhk range" in low
    assert "live in" in low
    assert "larger" not in low
    assert "smaller" not in low
    assert "what else would you like to know" not in low


def test_unsure_about_configuration_size_and_budget_moves_on():
    app_v3.conversation_memory.update({
        "stage": "qualify",
        "configuration_asked": True,
        "last_agent_question": "Which configuration are you looking for?",
    })
    spoken, _, stage = _turn("I am not sure.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["customer_configuration"] is None
    assert "configuration" in app_v3.conversation_memory["open_topics"]
    low = spoken.lower()
    assert "leave the configuration open" in low
    assert "live in" in low
    assert "which configuration" not in low
    assert "one, two, three" not in low

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "2 BHK",
        "last_agent_question": "Which of these sizes suits you better?",
        "last_spoken_reply": (
            "2 BHK Luxe is 1240 to 1248 sq. ft. 2 BHK Grande is 1337 to 1678 sq. ft. "
            "Which of these sizes suits you better?"
        ),
    })
    spoken, _, _ = _turn("I am not sure.")
    assert CALLS == []
    assert app_v3.conversation_memory["size_decided"] is False
    assert app_v3.conversation_memory["customer_variant"] is None
    assert "size" in app_v3.conversation_memory["open_topics"]
    low = spoken.lower()
    assert "leave the size open" in low
    assert "live in" in low
    assert "sq" not in low
    assert "larger" not in low

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "2 BHK",
        "last_agent_question": "What budget range should I keep in mind?",
    })
    spoken, _, _ = _turn("I am not sure.")
    assert CALLS == []
    assert "budget" in app_v3.conversation_memory["open_topics"]
    low = spoken.lower()
    assert "leave the budget for now" in low
    assert "location" in low
    assert "crore" not in low
    assert "price" not in low
    assert "spend" not in low


def test_investment_is_acknowledged_without_the_model():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "4 BHK",
        "customer_variant": "Grande",
        "size_decided": True,
        "covered_topics": ["location", "configuration"],
        "last_agent_question": "Are you looking at this to live in, or as an investment?",
    })
    spoken, _, stage = _turn("It's for an investment purpose.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["customer_purpose"] == "investment"
    low = spoken.lower()
    assert "investment" in low
    assert "electronic city" not in low
    assert "hosur road" not in low
    assert "kind of home" not in low
    assert "1.8 crore" in low
    assert "?" not in spoken


def test_a_location_statement_is_dropped_after_the_greeting():
    app_v3.conversation_memory["covered_topics"] = ["location"]
    spoken = app_v3.drop_restated_location(
        "SOBHA Townpark is located near Electronic City on Hosur Road, which is a well-connected area. "
        "Would you like to hear about the kind of home?"
    )
    low = spoken.lower()
    assert "electronic city" not in low
    assert "hosur road" not in low
    assert "kind of home" in low


def test_later_not_sure_does_not_ask_the_stored_size_again():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "4 BHK",
        "customer_variant": "Grande",
        "size_decided": True,
        "last_agent_question": "Would you like to know about the location?",
    })
    spoken, _, stage = _turn("I am not sure.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["size_decided"] is True
    low = spoken.lower()
    assert "sq" not in low
    assert "larger" not in low
    assert "grande" not in low
    assert "what else would you like to know" not in low
    assert "live in" in low or "kind of home" in low


def test_need_some_time_still_ends_the_call():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "2 BHK",
        "last_agent_question": "Which of these sizes suits you better?",
    })
    spoken, _, stage = _turn("I need some time.")
    assert stage == "closed"
    assert "take your time" in spoken.lower()
    assert CALLS == []


def test_open_configuration_is_not_asked_again():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "last_agent_question": "Are you looking at this to live in, or as an investment?",
    })
    REPLIES["To live in."] = (
        "Living in it makes sense. We have 1, 2, 3 and 4 BHK options here. "
        "Would you prefer a smaller or larger layout for your family?"
    )
    spoken, _, stage = _turn("To live in.")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "own use" in low
    assert "1.8 crore" in low
    assert "1, 2, 3" not in low
    assert "smaller or larger" not in low
    assert "which configuration" not in low
    assert "bhk" not in low
    assert "?" not in spoken

    app_v3.conversation_memory.update({
        "open_topics": ["configuration"],
        "customer_configuration": None,
        "customer_purpose": "live",
        "last_agent_question": "Would you prefer a smaller or larger layout for your family?",
    })
    REPLIES["Soba is fine."] = (
        "That works well. SOBHA Townpark offers 1 BHK to 4 BHK layouts. "
        "Which configuration fits your family best?"
    )
    spoken, _, _ = _turn("Soba is fine.")
    low = spoken.lower()
    assert "that works well" in low
    assert "1 bhk to 4 bhk" not in low
    assert "which configuration" not in low
    assert "configuration" in app_v3.conversation_memory["open_topics"]
    assert app_v3.conversation_memory["customer_configuration"] is None


def test_short_bhk_reply_is_stored_and_sizes_stay_canonical():
    assert app_v3.extract_configuration("one BHK") == "1 BHK"
    app_v3.conversation_memory["last_agent_question"] = "Which configuration are you looking for?"
    assert app_v3.extract_configuration("One to it.") == "1 BHK"
    app_v3.conversation_memory["last_agent_question"] = "Is now a good time?"
    assert app_v3.extract_configuration("One to it.") is None

    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "customer_purpose": "live",
        "last_agent_question": "Which configuration fits your family best?",
    })
    REPLIES["One to it."] = (
        "One BHK works well for a compact setup. "
        "Do you want to see the available sizes for that?"
    )
    spoken, _, _ = _turn("One to it.")
    assert app_v3.conversation_memory["customer_configuration"] == "1 BHK"
    assert "configuration" not in app_v3.conversation_memory["open_topics"]
    assert "one bhk works well" in spoken.lower()

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "customer_purpose": "live",
        "last_agent_question": "Would you like to know about the location?",
    })
    swapped = (
        "We have 1 BHK, 2 BHK, 3 BHK and 4 BHK apartments. "
        "The 1 BHK Luxe is 754 sq. ft., the 2 BHK ranges from 1240 to 1248 sq. ft. in Luxe "
        "and 1337 to 1678 sq. ft. in Grande, the 3 BHK ranges from 1506 to 1514 sq. ft. in Grande "
        "and 1337 to 1678 sq. ft. in Luxe, and the 4 BHK Grande ranges from 2203 to 2846 sq. ft. "
        "Which configuration interests you?"
    )
    REPLIES["tell me about floor plan and configurations"] = swapped
    fixed = app_v3.use_unit_sizes("tell me about floor plan and configurations", swapped)
    assert "3 BHK Luxe is 1506 to 1514 sq. ft." in fixed
    assert "3 BHK Grande is 1842 to 2547 sq. ft." in fixed
    assert "in Grande and 1337" not in fixed

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "last_agent_question": "Which configuration are you looking for?",
    })
    spoken, _, stage = _turn("BHK.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["customer_configuration"] is None
    low = spoken.lower()
    assert "one, two, three, or four bhk" in low
    assert "sq" not in low
    assert "1240" not in spoken


def test_spoken_reply_is_not_cut_mid_word():
    raw = (
        "We have 1 BHK, 2 BHK, 3 BHK and 4 BHK apartments. "
        "The 1 BHK Luxe is 754 sq. ft., the 2 BHK ranges from 1240 to 1248 sq. ft. in Luxe "
        "and 1337 to 1678 sq. ft. in Grande, the 3 BHK ranges from 1506 to 1514 sq. ft. in Grande "
        "and 1337 to 1678 sq. ft. in Luxe, and the 4 BHK Grande ranges from 2203 to 2846 sq. ft. "
        "Which configuration interests you?"
    )
    cleaned = app_v3.clean_response(raw)
    assert cleaned.endswith("Which configuration interests you?")
    assert not cleaned.endswith("Which confi")
    fitted = app_v3.fit_spoken_length(raw, limit=80)
    assert fitted.endswith(".")
    assert "confi" not in fitted


def test_not_interested_closes_and_bare_okay_does_not_book():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "last_agent_question": "Which configuration interests you?",
    })
    spoken, _, stage = _turn("Okay, okay, I'm not interested in this.")
    assert stage == "closed"
    assert "site visit" not in spoken.lower()
    assert "sales advisor" in spoken.lower()
    assert "have a good day" in spoken.lower()
    assert CALLS == []

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "last_agent_question": "Would you like to schedule a site visit to SOBHA Townpark?",
    })
    spoken, _, stage = _turn("Ah okay.")
    assert stage == "propose_slot"
    assert "which day" not in spoken.lower()
    assert "leave it for now" in spoken.lower()
    assert CALLS == []

    spoken, _, stage = _turn("Yes.")
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "have a good day" in spoken.lower()
    assert "which day" not in spoken.lower()
    assert "ten a m" not in spoken.lower()
    assert "booked for" not in spoken.lower()
    assert CALLS == []

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "visit_day",
        "last_agent_question": "Which day would suit you best for the visit?",
    })
    spoken, _, stage = _turn("Sunday.")
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "which day" not in spoken.lower()
    assert "ten a m" not in spoken.lower()
    assert "booked for" not in spoken.lower()
    assert CALLS == []


def test_greeting_location_is_not_offered_again():
    app_v3.remember_spoken(
        "Hi, this is Sobha Limited calling. I'm getting in touch about SOBHA Townpark - "
        "a New York-inspired luxury community near Electronic City on Hosur Road in Bengaluru. "
        "Is now a good time for a quick call?"
    )
    assert "location" in app_v3.conversation_memory["covered_topics"]
    assert "home" in app_v3.conversation_memory["covered_topics"]
    assert app_v3.conversation_memory["theme_spoken"] is True
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "customer_purpose": None,
        "last_agent_question": "Are you looking at this to live in, or as an investment?",
    })
    REPLIES["To live in."] = (
        "Living in is the perfect fit for SOBHA Townpark. "
        "Would you like to know about the location?"
    )
    spoken, _, stage = _turn("To live in.")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "own use" in low
    assert "1.8 crore" in low
    assert "location" not in low
    assert "kind of home" not in low
    assert "new york" not in low
    assert "configuration" not in low


def test_declined_location_does_not_return_to_configurations():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "customer_purpose": "live",
        "covered_topics": ["location"],
        "last_agent_question": "Would you like to know about the location?",
    })
    REPLIES["No, I you already told about that."] = (
        "Understood. Since you are already aware of that, "
        "shall we look at the available configurations?"
    )
    spoken, _, stage = _turn("No, I you already told about that.")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "configuration" not in low
    assert "location" not in low
    assert "kind of home" in low or "price" in low or "amenities" in low
    assert "location" in app_v3.conversation_memory["declined_topics"]

    filtered = app_v3.leave_closed_configuration(
        "No, I you already told about that.",
        "Understood. Since you are already aware of that, "
        "shall we look at the available configurations?",
    )
    assert "configuration" not in filtered.lower()


def test_not_interested_in_that_continues():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "customer_purpose": "live",
        "covered_topics": ["location"],
        "last_agent_question": "Shall we look at the available configurations?",
    })
    spoken, _, stage = _turn("No, I'm not interested in that.")
    assert stage == "answer_faq"
    assert CALLS == []
    low = spoken.lower()
    assert "thank you for your time" not in low
    assert "configuration" not in low
    assert "kind of home" in low or "price" in low or "amenities" in low
    assert "configuration" in app_v3.conversation_memory["declined_topics"]

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "last_agent_question": "Would you like to hear about the kind of home?",
    })
    spoken, _, stage = _turn("I'm not interested.")
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "have a good day" in spoken.lower()
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


def test_yes_to_kind_of_home_is_answered_without_the_model():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "customer_purpose": "investment",
        "covered_topics": ["location"],
        "last_agent_question": "Would you like to hear about the kind of home?",
    })
    REPLIES["Yeah, sure."] = "We have 1 to 4 BHK apartments. Which configuration are you looking for?"
    spoken, _, stage = _turn("Yeah, sure.")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "new york" in low
    assert "starting price" in low
    assert "kind of home" not in low
    assert "bhk" not in low
    assert "home" in app_v3.conversation_memory["covered_topics"]


def test_already_asked_and_a_no_leave_the_topic():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_purpose": "investment",
        "covered_topics": ["location", "home"],
        "last_agent_question": "Would you like to know the starting price?",
    })
    spoken, _, stage = _turn("You already asked that.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert "price" in app_v3.conversation_memory["declined_topics"]
    assert "kind of home" not in spoken.lower()
    assert "amenities" in spoken.lower()

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_purpose": "investment",
        "covered_topics": ["location", "home", "price"],
        "open_topics": ["configuration"],
        "last_agent_question": "Are you looking for a specific layout?",
    })
    spoken, _, stage = _turn("No, I am not.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert "bedroom" not in spoken.lower()
    assert "kind of home" not in spoken.lower()


def test_open_configuration_drops_the_menu_and_a_false_choice():
    app_v3.conversation_memory.update({
        "open_topics": ["configuration"],
        "customer_configuration": None,
    })
    menu = app_v3.leave_closed_configuration(
        "No, I am not.",
        "Understood. We have one to four bedroom apartments available. "
        "Would you like to hear about the amenities?",
    )
    assert "bedroom" not in menu.lower()
    assert "amenities" in menu.lower()
    claim = app_v3.drop_false_configuration_claim(
        "Understood. Since you have already chosen a configuration, "
        "I will share details specific to that. Would you like to hear about the amenities?"
    )
    assert "already chosen" not in claim.lower()
    assert "amenities" in claim.lower()


def test_not_finalized_leaves_configuration_open():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "last_agent_question": "Which configuration interests you?",
    })
    spoken, _, stage = _turn("I haven't finalized it.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert "configuration" in app_v3.conversation_memory["open_topics"]
    low = spoken.lower()
    assert "leave the configuration open" in low
    assert "investment" in low
    assert "bhk" not in low
    assert "sq" not in low
    assert "specific" not in low


def test_good_about_one_topic_can_still_offer_the_visit():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "3 BHK",
        "last_agent_question": "Which of these sizes suits you better?",
    })
    spoken, _, stage = _turn("No, I am good.")
    assert CALLS == []
    assert stage == "propose_slot"
    assert "site visit" in spoken.lower()
    assert "?" in spoken


def test_later_about_the_visit_closes_politely():
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "last_agent_question": "Would you like to schedule a site visit to SOBHA Townpark?",
    })
    spoken, _, stage = _turn("I will let you know later.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "highlights" not in spoken.lower()
    assert "?" not in spoken


def test_yes_to_highlights_arranges_details_and_does_not_book_a_day():
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "last_agent_question": "Would you like me to share the highlights of the project with you?",
    })
    spoken, _, stage = _turn("Yeah, you can share me.")
    assert CALLS == []
    assert stage == "closed"
    assert "arrange to share the details" in spoken.lower()
    assert "which day" not in spoken.lower()
    assert "?" not in spoken


def test_project_no_and_no_details_close_on_the_visit_day():
    app_v3.conversation_memory.update({
        "stage": "visit_day",
        "visit_offered": True,
        "last_agent_question": "Which day would suit you best for the visit?",
    })
    spoken, _, stage = _turn("No, I'm not interested.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "?" not in spoken

    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "visit_day",
        "visit_offered": True,
        "last_agent_question": "Which day would suit you best for the visit?",
    })
    spoken, _, stage = _turn("I don't need any details.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "?" not in spoken


def test_good_after_the_visit_was_offered_closes():
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "last_agent_question": "Would you like to schedule a site visit to SOBHA Townpark?",
    })
    spoken, _, stage = _turn("I'm good for now. Thank you.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "site visit" not in spoken.lower()


def test_invented_percent_and_specific_configuration_are_dropped():
    dropped = app_v3.drop_unsupported_percent(
        "It offers apartments with 67% of the area dedicated to open green spaces. "
        "Would you like to know the starting price?"
    )
    assert "67%" not in dropped
    assert "starting price" in dropped.lower()
    app_v3.conversation_memory.update({
        "open_topics": ["configuration"],
        "customer_configuration": None,
    })
    kept = app_v3.leave_closed_configuration(
        "I haven't finalized it.",
        "Would you like to know more about any specific configuration? "
        "Would you like to know the specific sizes for any of these?",
    )
    assert kept == ""


def test_a_yes_to_details_during_questions_closes():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "last_agent_question": "Would you like me to share the other available details with you?",
    })
    spoken, _, stage = _turn("Yeah, sure you can share me.")
    assert CALLS == []
    assert stage == "closed"
    low = spoken.lower()
    assert "arrange to share the details" in low
    assert "open space" not in low
    assert "?" not in spoken


def test_a_yes_to_the_price_still_states_the_price():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "last_agent_question": "Would you like to know the starting price?",
    })
    spoken, _, stage = _turn("Yes.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert "1.8 crore" in spoken.lower()
    assert "arrange to share" not in spoken.lower()


def test_a_second_what_else_offers_the_visit_once():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_purpose": "investment",
        "covered_topics": ["location", "home", "price", "amenities"],
        "fallback_asked": True,
        "visit_offered": False,
        "last_agent_question": "Would you like to know about the open space allocation?",
    })
    spoken, _, stage = _turn("No, I'm not interested in that.")
    assert CALLS == []
    assert stage == "propose_slot"
    assert "site visit" in spoken.lower()
    assert "what else can i help you with" not in spoken.lower()


def test_a_lone_bhk_during_the_greeting_is_not_a_choice():
    app_v3.conversation_memory["stage"] = "greeting"
    spoken, _, stage = _turn("BHK.")
    assert CALLS == []
    assert stage == "greeting"
    assert app_v3.conversation_memory["customer_configuration"] is None
    assert "good time" in spoken.lower()
    assert "sq" not in spoken.lower()
    assert "1240" not in spoken


def test_a_bare_bhk_asks_for_a_number():
    app_v3.conversation_memory.update({
        "stage": "qualify",
        "configuration_asked": True,
        "last_agent_question": "Which configuration are you looking for?",
    })
    spoken, _, stage = _turn("BHK.")
    assert CALLS == []
    assert stage == "qualify"
    assert app_v3.conversation_memory["customer_configuration"] is None
    assert spoken.strip() == "One, two, three, or four BHK?"
    assert "sq" not in spoken.lower()

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "qualify",
        "configuration_asked": True,
        "last_agent_question": "Which configuration are you looking for?",
    })
    spoken, _, _ = _turn("configuration.")
    assert CALLS == []
    assert spoken.strip() == "One, two, three, or four BHK?"

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "qualify",
        "configuration_asked": True,
        "last_agent_question": "Which configuration are you looking for?",
    })
    spoken, _, _ = _turn("2 BHK.")
    assert app_v3.conversation_memory["customer_configuration"] == "2 BHK"
    assert "one, two, three, or four" not in spoken.lower()


def test_investment_after_the_greeting_asks_the_price():
    app_v3.remember_spoken(
        "Hi, this is Sobha Limited calling. I'm getting in touch about SOBHA Townpark - "
        "a New York-inspired luxury community near Electronic City on Hosur Road in Bengaluru. "
        "Is now a good time for a quick call?"
    )
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_topics": ["configuration"],
        "last_agent_question": "Are you looking at this to live in, or as an investment?",
    })
    spoken, _, stage = _turn("It's for an investment purpose.")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "investment" in low
    assert "1.8 crore" in low
    assert "kind of home" not in low
    assert "new york" not in low
    assert "?" not in spoken

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "covered_topics": ["location", "home", "price"],
        "price_stated": True,
        "last_agent_question": "Are you looking at this to live in, or as an investment?",
    })
    spoken, _, _ = _turn("It's for an investment purpose.")
    assert CALLS == []
    assert "anything else about the project" in spoken.lower()
    assert "amenities" not in spoken.lower()


def test_a_yes_after_the_theme_does_not_say_new_york_again():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "theme_spoken": True,
        "covered_topics": ["location", "home"],
        "customer_purpose": "investment",
        "last_agent_question": "Would you like to hear about the kind of home?",
    })
    spoken, _, stage = _turn("Yeah, what does it mean?")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "new york" not in low
    assert "starting price" in low
    assert spoken.count("?") == 1


def test_a_new_york_sentence_is_dropped_after_the_greeting():
    app_v3.conversation_memory["theme_spoken"] = True
    spoken = app_v3.drop_restated_theme(
        "SOBHA Townpark is a New York-themed luxury community. "
        "Would you like to know the starting price?"
    )
    low = spoken.lower()
    assert "new york" not in low
    assert "starting price" in low


def test_okay_after_the_price_asks_the_open_question():
    for speech in ("Okay.", "Okay, that's good.", "That's good.", "Sounds good."):
        app_v3.reset_memory()
        CALLS.clear()
        app_v3.conversation_memory.update({
            "stage": "answer_faq",
            "price_stated": True,
            "covered_topics": ["price"],
            "last_agent_question": None,
        })
        spoken, _, stage = _turn(speech)
        assert CALLS == []
        assert stage == "answer_faq"
        low = spoken.lower()
        assert "anything else about the project" in low
        assert "amenities" not in low
        assert "flexible payment" not in low
        assert "home loan" not in low

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({"stage": "answer_faq", "price_stated": True})
    spoken, _, stage = _turn("What about the payment plans?")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "flexible payment plans" in low
    assert "home loan" in low
    assert "anything else about the project" in low


def test_a_lower_budget_offers_a_sales_callback():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "price_stated": True,
    })
    spoken, _, stage = _turn("It's too expensive and I am looking for a budget of 1.5 Cr or below.")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "other projects" in low
    assert "call back" in low
    assert "hamptons" not in low
    assert "madison" not in low

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "awaiting_callback": True,
    })
    spoken, _, stage = _turn("Yes.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "awaiting_callback": True,
    })
    spoken, _, stage = _turn("No.")
    assert CALLS == []
    assert stage == "propose_slot"
    assert "site visit" in spoken.lower()


def test_a_no_to_anything_else_offers_the_visit():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_project_asked": True,
        "last_agent_question": "Would you like to know anything else about the project?",
    })
    spoken, _, stage = _turn("No.")
    assert CALLS == []
    assert stage == "propose_slot"
    assert "site visit" in spoken.lower()

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "last_agent_question": "Would you like to hear about the amenities?",
    })
    spoken, _, stage = _turn("No, I am not interested to know about them.")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "anything else about the project" in low
    assert "what else can i help you with" not in low
    assert "sales advisor" not in low


def test_unsure_after_the_open_question_offers_the_visit():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "open_project_asked": True,
        "customer_purpose": "investment",
        "covered_topics": ["location", "home", "price", "amenities"],
        "last_agent_question": "Would you like to know anything else about the project?",
    })
    spoken, _, stage = _turn("I'm not sure what to ask.")
    assert CALLS == []
    assert stage == "propose_slot"
    low = spoken.lower()
    assert "site visit" in low
    assert "what else can i help you with" not in low
    assert "cannot book" not in low

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "fallback_asked": True,
        "customer_purpose": "investment",
        "covered_topics": ["location", "home", "price", "amenities"],
        "last_agent_question": "What else would you like to know?",
    })
    spoken, _, stage = _turn("I'm not sure about this.")
    assert CALLS == []
    assert stage == "propose_slot"
    low = spoken.lower()
    assert "site visit" in low
    assert "what else can i help you with" not in low


def test_a_question_about_the_visit_offers_the_visit():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "price_stated": True,
        "open_project_asked": True,
        "last_agent_question": "Would you like to know anything else about the project?",
    })
    spoken, _, stage = _turn("Can you let me know about the visit?")
    assert CALLS == []
    assert stage == "propose_slot"
    low = spoken.lower()
    assert "site visit" in low
    assert "which day" not in low
    assert "cannot book" not in low
    assert "share the" not in low

    spoken, _, stage = _turn("Yes.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "have a good day" in spoken.lower()
    assert "which day" not in spoken.lower()
    assert "ten a m" not in spoken.lower()

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({"stage": "answer_faq"})
    spoken, _, stage = _turn("I want to book a visit.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "which day" not in spoken.lower()
    assert "ten a m" not in spoken.lower()


def test_come_back_later_closes_the_visit():
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "last_agent_question": "Would you like to schedule a site visit to SOBHA Townpark?",
    })
    spoken, _, stage = _turn("I will come back to you later.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "?" not in spoken


def test_a_refusal_of_details_closes():
    app_v3.conversation_memory.update({
        "stage": "propose_slot",
        "visit_offered": True,
        "last_agent_question": "Would you like me to share the available details for later?",
    })
    spoken, _, stage = _turn("No, I don't want any details.")
    assert CALLS == []
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "?" not in spoken


def test_any_configuration_stays_open():
    app_v3.conversation_memory.update({
        "stage": "qualify",
        "configuration_asked": True,
        "last_agent_question": "Which configuration are you looking for?",
    })
    spoken, _, _stage = _turn("I am okay with any configuration.")
    assert CALLS == []
    low = spoken.lower()
    assert "leave the configuration open" in low
    assert "754" not in low
    assert "1240" not in low


def test_any_configuration_and_the_project_states_the_price():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "configuration_asked": True,
        "open_topics": ["configuration"],
        "last_agent_question": "Which one interests you?",
    })
    spoken, _, _stage = _turn(
        "Again, I am okay with any configuration. You can let me know about the project."
    )
    assert CALLS == []
    low = spoken.lower()
    assert "leave the configuration open" in low
    assert "1.8" in low
    assert "754" not in low
    assert "1240" not in low


def test_a_layout_question_does_not_invent_a_floor_plan():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "1 BHK",
        "last_spoken_reply": "The 1 BHK Luxe is 754 sq. ft. Would you like to know more about the layout?",
        "last_agent_question": "Would you like to know more about the layout?",
    })
    spoken, _, _stage = _turn("What's the layout?")
    assert CALLS == []
    low = spoken.lower()
    assert "floor plan" in low
    assert "zoning" not in low
    assert "joining" not in low
    assert "754" not in spoken

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "1 BHK",
        "last_spoken_reply": "Would you like to know about the internal zoning?",
        "last_agent_question": "Would you like to know about the internal zoning?",
    })
    spoken, _, _stage = _turn("What does internal joining mean?")
    assert CALLS == []
    low = spoken.lower()
    assert "floor plan" in low
    assert "754" in spoken
    assert "zoning" not in low
    assert "seamless" not in low


def test_a_model_layout_question_is_dropped():
    spoken = app_v3.drop_invented_floor_plan(
        "The 1 BHK Luxe is 754 sq. ft. Would you like to know about the internal zoning?"
    )
    assert "754" in spoken
    assert "zoning" not in spoken.lower()
    assert "?" not in spoken


def test_to_live_in_is_a_purpose_not_an_amenity():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "3 BHK",
        "last_agent_question": "Are you looking at this to live in, or as an investment?",
    })
    spoken, _, stage = _turn("It's to live in.")
    assert CALLS == []
    assert stage == "answer_faq"
    assert app_v3.conversation_memory["customer_purpose"] == "live"
    low = spoken.lower()
    assert "own use" in low
    assert "1.8 crore" in low
    assert "open space" not in low
    assert "clubhouse" not in low
    assert "?" not in spoken


def test_a_worded_percent_is_dropped_unless_the_source_has_it():
    dropped = app_v3.drop_unsupported_percent(
        "Sixty seven percent of the total area is open space. "
        "Would you like to know the starting price?"
    )
    assert "percent" not in dropped.lower()
    assert "starting price" in dropped.lower()
    kept = app_v3.drop_unsupported_percent(
        "Sixty seven percent of the total area is open space.",
        "The brochure says 67% of the land is open.",
    )
    assert "sixty seven percent" in kept.lower()


def test_an_unasked_clubhouse_question_is_replaced():
    spoken = app_v3.drop_unasked_amenity_question(
        "It's to live in.",
        "Noted. Would you like to know about the clubhouse and swimming pool facilities?",
    )
    low = spoken.lower()
    assert "clubhouse" not in low
    assert "swimming" not in low
    assert "anything else about the project" in low


def test_decide_later_leaves_the_size_open():
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "customer_configuration": "3 BHK",
        "last_spoken_reply": (
            "For 3 BHK, Luxe is 1506 to 1514 sq. ft. and Grande is 1842 to 2547 sq. ft. "
            "Which one interests you?"
        ),
        "last_agent_question": "Which one interests you?",
    })
    spoken, _, stage = _turn("I will decide this later after seeing.")
    assert CALLS == []
    assert stage == "answer_faq"
    low = spoken.lower()
    assert "leave the size open" in low
    assert "not listed" not in low
    assert "live in" in low


def test_a_denial_of_listed_sizes_is_replaced():
    app_v3.conversation_memory["customer_configuration"] = "3 BHK"
    spoken = app_v3.restore_denied_sizes(
        "Noted. For the 3 BHK, the available sizes are not listed here. "
        "Would you like me to share the other details?"
    )
    low = spoken.lower()
    assert "not listed" not in low
    assert "1506" in spoken
    assert "1842" in spoken
    assert "2547" in spoken
    assert "share" not in low
    assert "anything else about the project" in low


def test_a_saturday_sentence_is_confirmed_as_the_day():
    app_v3.conversation_memory.update({
        "stage": "visit_day",
        "visit_offered": True,
        "last_agent_question": "Which day would suit you best for the visit?",
    })
    spoken, _, stage = _turn("This Saturday would be fine.")
    assert CALLS == []
    assert stage == "closed"
    low = spoken.lower()
    assert "sales advisor" in low
    assert "have a good day" in low
    assert "which day" not in low
    assert "ten a m" not in low
    assert "booked for" not in low

    app_v3.reset_memory()
    CALLS.clear()
    app_v3.conversation_memory.update({
        "stage": "visit_pick_slot",
        "visit_offered": True,
        "last_agent_question": "Which one works best for you?",
    })
    spoken, _, stage = _turn("4 p.m. would be fine.")
    assert CALLS == []
    assert stage == "closed"
    low = spoken.lower()
    assert "sales advisor" in low
    assert "ten a m" not in low
    assert "booked for" not in low
    assert "SOBHA" in app_v3.visit_schedule_question()


_PURPOSE = {
    "customer_configuration": "2 BHK",
    "configuration_asked": True,
    "last_agent_question": "Are you looking at this to live in, or as an investment?",
}
_VISIT = {
    "visit_offered": True,
    "last_agent_question": "Would you like to schedule a site visit to SOBHA Townpark?",
}


@pytest.mark.parametrize(
    ("stage", "speech", "prepare", "expect_stage", "needle", "marker"),
    [
        ("greeting", "Yes.", {}, "qualify", "one, two, three", ""),
        ("greeting", "Not now.", {}, "closed", "lovely day", ""),
        (
            "greeting",
            "Hmm.",
            {"last_agent_question": "Is now a good time for a quick call?"},
            "greeting",
            "good time",
            "",
        ),
        (
            "greeting",
            "Maybe.",
            {"last_agent_question": "Is now a good time for a quick call?"},
            "greeting",
            "good time",
            "",
        ),
        ("qualify", "Sounds good.", _PURPOSE, "answer_faq", "location", "purpose-kept"),
        ("qualify", "No.", _PURPOSE, "answer_faq", "location", "purpose-declined"),
        ("qualify", "Hmm.", _PURPOSE, "answer_faq", "location", "purpose-kept"),
        ("qualify", "Maybe.", _PURPOSE, "answer_faq", "location", "purpose-open"),
        (
            "answer_faq",
            "Sounds good.",
            {
                "customer_configuration": "2 BHK",
                "last_agent_question": "Would you like to know about the location?",
            },
            "answer_faq",
            "electronic city",
            "",
        ),
        ("answer_faq", "No.", _PURPOSE, "answer_faq", "what would you like to know next", ""),
        ("answer_faq", "I see.", _PURPOSE, "answer_faq", "location", "purpose-kept"),
        ("answer_faq", "Maybe.", _PURPOSE, "answer_faq", "location", "purpose-open"),
        ("propose_slot", "Yes.", _VISIT, "closed", "sales advisor", ""),
        ("propose_slot", "No.", _VISIT, "closed", "sales advisor", ""),
        ("propose_slot", "Hmm.", _VISIT, "propose_slot", "leave it for now", ""),
        ("propose_slot", "Maybe.", _VISIT, "closed", "sales advisor", ""),
    ],
)
def test_reply_kind_covers_each_stage(stage, speech, prepare, expect_stage, needle, marker):
    app_v3.conversation_memory.update({"stage": stage, **prepare})
    spoken, _, got = _turn(speech)
    assert CALLS == []
    assert got == expect_stage
    assert needle in spoken.lower()
    if marker == "purpose-open":
        assert "purpose" in app_v3.conversation_memory["open_topics"]
    elif marker == "purpose-declined":
        assert "purpose" in app_v3.conversation_memory["declined_topics"]
    elif marker == "purpose-kept":
        assert "purpose" not in app_v3.conversation_memory["open_topics"]
