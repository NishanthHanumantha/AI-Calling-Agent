"""Details the card does not store stay on a written line."""

import asyncio

import pytest

import app_v3


class _Request:
    def __init__(self, speech):
        self._form = {"SpeechResult": speech, "UnstableSpeechResult": ""}

    async def form(self):
        return self._form


def _spoken(xml):
    import re

    parts = re.findall(r"<Say[^>]*>(.*?)</Say>", xml, flags=re.S)
    text = " ".join(part.strip() for part in parts if part.strip())
    return (
        text.replace("&apos;", "'")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def _turn(speech):
    response = asyncio.run(app_v3.handle_speech(_Request(speech), SpeechResult=speech))
    xml = response.body.decode() if isinstance(response.body, bytes) else response.body
    return _spoken(xml), app_v3.conversation_memory["stage"]


def _forbid_chat(monkeypatch):
    def post(url, *args, **kwargs):
        if "chat/completions" in str(url):
            raise AssertionError("the model was called")
        raise AssertionError(url)

    monkeypatch.setattr(app_v3.requests, "post", post)


def _birdsong():
    app_v3.reset_memory()
    project = next(
        item for item in app_v3.catalog.PROJECTS if "birdsong" in item["spoken"].lower()
    )
    app_v3.conversation_memory.update(
        {
            "stage": "answer_faq",
            "catalog_flow": True,
            "catalog_done": True,
            "active_project_ids": [project["id"]],
            "customer_part": "north",
            "customer_purpose": "live",
            "last_agent_question": "Would you like to know anything else about the project?",
        }
    )
    return project


@pytest.mark.parametrize(
    "speech",
    [
        "How much is the maintenance charge?",
        "Is this project RERA registered?",
        "What is the car parking arrangement?",
        "What is the possession timeline?",
        "Please tell me the approval status.",
        "Please tell me who the builder is.",
    ],
)
def test_an_unstored_detail_stays_off_the_price_and_the_model(monkeypatch, speech):
    _forbid_chat(monkeypatch)
    _birdsong()
    spoken, stage = _turn(speech)
    low = spoken.lower()
    assert "not confirmed" in low
    assert "sales advisor" in low
    assert "crore" not in low
    assert "what would you like to know" not in low
    assert stage == "answer_faq"


def test_a_price_question_still_uses_the_card(monkeypatch):
    _forbid_chat(monkeypatch)
    _birdsong()
    spoken, _stage = _turn("What is the price?")
    assert "1.2" in spoken
    assert "crore" in spoken.lower()
    assert "not confirmed" not in spoken.lower()


def test_a_hindi_request_repeats_the_last_question(monkeypatch):
    _forbid_chat(monkeypatch)
    _birdsong()
    question = "Would you like to know anything else about the project?"
    spoken, stage = _turn("Uh yeah, can you explain this in Hindi?")
    assert question in spoken
    assert app_v3.conversation_memory["customer_language"] == "hi-IN"
    assert stage == "answer_faq"


def test_a_deferred_visit_closes_with_the_advisor(monkeypatch):
    _forbid_chat(monkeypatch)
    _birdsong()
    app_v3.conversation_memory["visit_offered"] = True
    app_v3.conversation_memory["stage"] = "propose_slot"
    app_v3.conversation_memory["last_agent_question"] = (
        "Would you like to schedule a site visit to Chartered Birdsong?"
    )
    spoken, stage = _turn("I will check and tell you.")
    low = spoken.lower()
    assert stage == "closed"
    assert "sales advisor" in low
    assert "plot size" not in low
    assert "933" not in spoken
