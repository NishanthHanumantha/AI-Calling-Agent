"""A recognized catalog line stays written. Anything else can call the model."""

import asyncio

import app_v3


class _Request:
    def __init__(self, speech):
        self._form = {"SpeechResult": speech, "UnstableSpeechResult": ""}

    async def form(self):
        return self._form


class _Chat:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {"content": "The sales advisor can confirm that."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }


def _spoken(xml):
    import re

    parts = re.findall(r"<Say[^>]*>(.*?)</Say>", xml, flags=re.S)
    text = " ".join(part.strip() for part in parts if part.strip())
    return (
        text.replace("&apos;", "'")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
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


def test_a_costly_magnus_offer_names_the_cheaper_projects(monkeypatch):
    _forbid_chat(monkeypatch)
    app_v3.reset_memory()
    app_v3.conversation_memory.update({
        "stage": "qualify",
        "catalog_flow": True,
        "part_asked": True,
        "config_prompted": True,
        "purpose_prompted": True,
        "customer_part": "South",
        "customer_purpose": "investment",
        "customer_bedrooms": 3,
        "customer_configuration": "3 BHK",
        "last_agent_question": (
            "Okay, that's good. Are you looking closer to Electronic City, "
            "Bannerghatta, or Banashankari?"
        ),
    })
    spoken, stage = _turn("I'm looking closer to Bannerghatta.")
    assert stage == "qualify"
    assert "magnus" in spoken.lower()
    assert "3.4" in spoken

    spoken, stage = _turn("Uh, it's very costly. Can you suggest any cheaper projects?")
    low = spoken.lower()
    assert "townpark" in low
    assert "valley view" in low
    assert "eco-luxe" not in low
    assert "3.4" not in spoken
    assert "site visit" not in low
    assert stage == "qualify"

    spoken, stage = _turn("No, I'm good.")
    assert stage == "closed"
    assert "sales advisor" in spoken.lower()
    assert "site visit" not in spoken.lower()


def test_an_unrecognized_question_calls_the_model(monkeypatch):
    seen = []

    def post(url, *args, **kwargs):
        seen.append(str(url))
        if "chat/completions" not in str(url):
            raise AssertionError(url)
        return _Chat()

    monkeypatch.setattr(app_v3.requests, "post", post)
    app_v3.reset_memory()
    project = next(item for item in app_v3.catalog.PROJECTS if "birdsong" in item["spoken"].lower())
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "catalog_flow": True,
        "catalog_done": True,
        "active_project_ids": [project["id"]],
        "customer_part": "north",
        "customer_purpose": "live",
        "last_agent_question": "Would you like to know anything else about the project?",
    })
    _turn("What is the vaastu of these homes?")
    assert any("chat/completions" in url for url in seen)


def test_a_price_question_stays_on_the_card(monkeypatch):
    _forbid_chat(monkeypatch)
    app_v3.reset_memory()
    project = next(item for item in app_v3.catalog.PROJECTS if "birdsong" in item["spoken"].lower())
    app_v3.conversation_memory.update({
        "stage": "answer_faq",
        "catalog_flow": True,
        "catalog_done": True,
        "active_project_ids": [project["id"]],
        "last_agent_question": "Would you like to know anything else about the project?",
    })
    spoken, _stage = _turn("What is the price?")
    assert "1.2" in spoken
    assert "crore" in spoken.lower()
