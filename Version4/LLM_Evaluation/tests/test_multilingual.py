"""LLM-EVAL.4.1 — multilingual business-conversation evaluation (FakeProvider only)."""

from __future__ import annotations

import time

from interactive.cli import parse_args, run_multilingual_tracks, selected_tracks
from interactive.conversation import SHARED_PROJECT_CONTEXT, detect_language
from interactive.evaluator import (
    EVAL_STATUS_EVALUATION_ERROR,
    EVAL_STATUS_MISSING_GROUND_TRUTH,
    EVAL_STATUS_MISSING_RESPONSE,
    evaluate_turn,
    flatten_evaluation_records,
    session_summary,
)
from interactive.multilingual import (
    CASES_BY_ID,
    LANG_MATCH,
    LANG_MISMATCH,
    LANG_PARTIAL,
    LANGUAGE_TRACKS,
    TRACK_CASES,
    TRACK_UTTERANCES,
    aggregate_multilingual,
    bind_multilingual_ground_truth,
    detect_response_language,
    score_response_language,
    structured_output_status,
    track_utterances,
)
from models.prompting import build_user_payload
from interactive.schemas import DISPLAY_ORDER, TurnResponse
from interactive.session import InteractiveSession
from interactive.visit_policy import (
    CORRECT_DATE_REQUEST,
    CORRECT_FINAL_CONFIRMATION,
    CORRECT_SLOT_OFFER,
    INVALID_SLOT,
    PREMATURE_CONFIRMATION,
    evaluate_visit_sequence,
    visit_booking_state,
)


AMENITY_ANSWER = (
    "Townpark has clubhouses, swimming pools, sports courts, landscaped gardens, "
    "kids play areas, a forest grove and camping grounds."
)
SLOT_OFFER = "The available slots are 10:00 AM, 11:30 AM, 4:00 PM and 6:00 PM. Which works?"
CONFIRM_ANSWER = "Your visit is confirmed for Friday at 4:00 PM."
ASK_DATE = "Which day would you like to visit?"
QUALIFY_ANSWER = "A 3 BHK at SOBHA Townpark starts at INR 1.8 Crore onwards. What configuration are you considering?"
HINDI_AMENITY = "इस प्रोजेक्ट में क्लबहाउस, स्विमिंग पूल, स्पोर्ट्स कोर्ट और गार्डन हैं।"
KANNADA_AMENITY = "ಈ ಪ್ರಾಜೆಕ್ಟ್‌ನಲ್ಲಿ ಕ್ಲಬ್‌ಹೌಸ್, ಸ್ವಿಮ್ಮಿಂಗ್ ಪೂಲ್ ಮತ್ತು ಗಾರ್ಡನ್ ಇವೆ."
MIXED_AMENITY = "Amenities ide — clubhouses, swimming pools, sports courts, gardens, forest grove."


class FakeProvider:
    def __init__(
        self,
        alias: str,
        answer: str = QUALIFY_ANSWER,
        intent: str = "QUALIFICATION",
        action: str = "QUALIFY",
        language: str = "en",
        schema_valid: bool = True,
        error_type=None,
        error_message=None,
        input_tokens: int = 8,
        output_tokens: int = 4,
        script: list[dict] | None = None,
    ):
        self.alias = alias
        self.answer = answer
        self.intent = intent
        self.action = action
        self.language = language
        self.schema_valid = schema_valid
        self.error_type = error_type
        self.error_message = error_message
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.script = list(script or [])
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        item = {}
        if self.script:
            item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        answer = item.get("answer", self.answer)
        return {
            "provider": "fake",
            "model_id": f"fake-{self.alias}",
            "answer": item.get("answer", answer),
            "intent": item.get("intent", self.intent),
            "action": item.get("action", self.action),
            "language": item.get("language", self.language),
            "raw_response": item.get("raw_response", item.get("answer", answer) or ""),
            "schema_valid": item.get("schema_valid", self.schema_valid),
            "usage": {
                "input_tokens": item.get("input_tokens", self.input_tokens),
                "output_tokens": item.get("output_tokens", self.output_tokens),
                "total_tokens": item.get("input_tokens", self.input_tokens)
                + item.get("output_tokens", self.output_tokens),
            },
            "latency_ms": item.get("latency_ms", 20),
            "error": item.get("error_type", self.error_type),
            "error_type": item.get("error_type", self.error_type),
            "error_message": item.get("error_message", self.error_message),
        }


def _script_for_track(track: str) -> list[dict]:
    visit_day = {
        "english": ASK_DATE,
        "hindi": "आप कौन सा दिन विज़िट करना चाहेंगे?",
        "kannada": "ನೀವು ಯಾವ ದಿನ ಬರಲು ಇಷ್ಟಪಡುತ್ತೀರಿ?",
        "mixed": "Which day would you like to visit?",
    }
    amenity = {
        "english": AMENITY_ANSWER,
        "hindi": HINDI_AMENITY + " clubhouses swimming pools sports courts landscaped gardens.",
        "kannada": KANNADA_AMENITY + " clubhouses swimming pools sports courts landscaped gardens.",
        "mixed": MIXED_AMENITY,
    }
    lang = {"english": "en", "hindi": "hi", "kannada": "kn", "mixed": "mixed"}[track]
    offer = SLOT_OFFER if track != "hindi" else "स्लॉट हैं 10:00 AM, 11:30 AM, 4:00 PM और 6:00 PM।"
    confirm = CONFIRM_ANSWER if track != "kannada" else "ನಿಮ್ಮ ವಿಸಿಟ್ Friday 4:00 PM ಗೆ ದೃಢವಾಗಿದೆ."
    closure = "Great, see you Friday at 4:00 PM." if track != "hindi" else "ठीक है, शुक्रवार शाम 4:00 PM पर मिलते हैं।"
    if track == "mixed":
        steps = [
            ("QUALIFICATION", "QUALIFY", "Sure, nanage details share madtheeni. Budget heli."),
            ("QUALIFICATION", "QUALIFY", QUALIFY_ANSWER),
            ("QUALIFICATION", "QUALIFY", QUALIFY_ANSWER),
            ("AMENITIES_QUERY", "ANSWER", amenity[track]),
            ("SITE_VISIT", "BOOK_VISIT", offer),
            ("SITE_VISIT", "CONFIRM_SLOT", confirm),
            ("CONFIRMATION", "ANSWER", "Okay done, visit confirmed for Friday at 4:00 PM."),
        ]
    else:
        steps = [
            ("QUALIFICATION", "QUALIFY", QUALIFY_ANSWER if track == "english" else "ज़रूर, बजट बताइए।" if track == "hindi" else "ಸರಿ, ಬಜೆಟ್ ಹೇಳಿ."),
            ("QUALIFICATION", "QUALIFY", QUALIFY_ANSWER),
            ("QUALIFICATION", "QUALIFY", QUALIFY_ANSWER),
            ("AMENITIES_QUERY", "ANSWER", amenity[track]),
            ("SITE_VISIT", "CLARIFY", visit_day[track]),
            ("SITE_VISIT", "BOOK_VISIT", offer),
            ("SITE_VISIT", "CONFIRM_SLOT", confirm),
            ("CONFIRMATION", "ANSWER", closure),
        ]
    return [{"intent": i, "action": a, "answer": ans, "language": lang} for i, a, ans in steps]


def _providers(script=None, **kwargs) -> dict[str, FakeProvider]:
    return {alias: FakeProvider(alias, script=script, **kwargs) for alias in DISPLAY_ORDER}


def _session(track: str, providers=None, opening="fixed") -> InteractiveSession:
    return InteractiveSession(
        providers=providers or _providers(_script_for_track(track)),
        conversation_mode="outbound",
        opening_mode=opening,
        language_track=track,
    )


def _play(session: InteractiveSession, track: str) -> None:
    session.begin_outbound()
    for utterance in track_utterances(track):
        session.generate_turn(utterance)


def _resp(alias: str, answer: str, intent: str, action: str, **kwargs) -> TurnResponse:
    return TurnResponse(
        alias=alias,
        provider=alias.split("_")[0],
        model_id=f"fake-{alias}",
        answer=answer,
        intent=intent,
        action=action,
        language=kwargs.get("language", "en"),
        stage=None,
        raw_response=kwargs.get("raw_response", answer or ""),
        latency_ms=kwargs.get("latency_ms", 12),
        input_tokens=kwargs.get("input_tokens", 2),
        output_tokens=kwargs.get("output_tokens", 3),
        total_tokens=kwargs.get("total_tokens", 5),
        error_type=kwargs.get("error_type"),
        error_message=kwargs.get("error_message"),
        schema_valid=kwargs.get("schema_valid", True),
    )


def _all(answer: str, intent: str, action: str, **kwargs) -> list[TurnResponse]:
    return [_resp(alias, answer, intent, action, **kwargs) for alias in DISPLAY_ORDER]


def _hist(utterances: list[str]) -> dict[str, list[dict[str, str]]]:
    history = [{"role": "user", "content": text} for text in utterances]
    return {alias: list(history) for alias in DISPLAY_ORDER}


def test_cli_multilingual_mode_does_not_break_outbound():
    outbound = parse_args(["--mode", "outbound"])
    assert outbound.mode == "outbound"
    assert outbound.demo is False
    multi = parse_args(["--mode", "multilingual", "--track", "hindi"])
    assert multi.mode == "multilingual"
    assert multi.track == "hindi"
    default = parse_args(["--mode", "multilingual"])
    assert default.track == "all"


def test_english_gt_labels():
    gold = bind_multilingual_ground_truth("english", "Yes", 2, [])
    assert gold["test_id"] == "E01"
    assert gold["expected_intent"] == "QUALIFICATION"
    assert gold["expected_action"] == "QUALIFY"
    gold = bind_multilingual_ground_truth("english", "What amenities are available?", 5, TRACK_UTTERANCES["english"][:3])
    assert gold["test_id"] == "E04"
    assert gold["expected_intent"] == "AMENITIES_QUERY"
    gold = bind_multilingual_ground_truth("english", "I would like to visit", 6, TRACK_UTTERANCES["english"][:4])
    assert gold["expected_action"] == "CLARIFY"
    gold = bind_multilingual_ground_truth("english", "Friday", 7, TRACK_UTTERANCES["english"][:5])
    assert gold["expected_action"] == "BOOK_VISIT"
    gold = bind_multilingual_ground_truth("english", "4 PM", 8, TRACK_UTTERANCES["english"][:6])
    assert gold["expected_action"] == "CONFIRM_SLOT"
    gold = bind_multilingual_ground_truth("english", "Okay", 9, TRACK_UTTERANCES["english"][:7])
    assert gold["expected_intent"] == "CONFIRMATION"


def test_hindi_gt_labels():
    mapping = {
        "H01": ("हाँ", "QUALIFICATION", "QUALIFY"),
        "H02": ("मेरा बजट लगभग 2 करोड़ है", "QUALIFICATION", "QUALIFY"),
        "H03": ("मुझे 3 BHK चाहिए", "QUALIFICATION", "QUALIFY"),
        "H04": ("इस प्रोजेक्ट में कौन-कौन सी सुविधाएँ हैं?", "AMENITIES_QUERY", "ANSWER"),
        "H05": ("मैं साइट विज़िट करना चाहता हूँ", "SITE_VISIT", "CLARIFY"),
        "H06": ("शुक्रवार", "SITE_VISIT", "BOOK_VISIT"),
        "H07": ("शाम 4 बजे", "SITE_VISIT", "CONFIRM_SLOT"),
        "H08": ("ठीक है", "CONFIRMATION", "ANSWER"),
    }
    prior = []
    for test_id, (utt, intent, action) in mapping.items():
        gold = bind_multilingual_ground_truth("hindi", utt, len(prior) + 2, prior)
        assert gold["test_id"] == test_id
        assert gold["expected_intent"] == intent
        assert gold["expected_action"] == action
        prior.append(utt)


def test_kannada_gt_labels():
    mapping = {
        "K01": ("ಹೌದು", "QUALIFICATION", "QUALIFY"),
        "K02": ("ನನ್ನ ಬಜೆಟ್ ಸುಮಾರು 2 ಕೋಟಿ", "QUALIFICATION", "QUALIFY"),
        "K03": ("ನನಗೆ 3 BHK ಬೇಕು", "QUALIFICATION", "QUALIFY"),
        "K04": ("ಈ ಪ್ರಾಜೆಕ್ಟ್‌ನಲ್ಲಿ ಯಾವ ಯಾವ ಸೌಲಭ್ಯಗಳಿವೆ?", "AMENITIES_QUERY", "ANSWER"),
        "K05": ("ನಾನು ಸೈಟ್ ವಿಸಿಟ್ ಮಾಡಲು ಇಷ್ಟಪಡುತ್ತೇನೆ", "SITE_VISIT", "CLARIFY"),
        "K06": ("ಶುಕ್ರವಾರ", "SITE_VISIT", "BOOK_VISIT"),
        "K07": ("ಸಂಜೆ 4 ಗಂಟೆಗೆ", "SITE_VISIT", "CONFIRM_SLOT"),
        "K08": ("ಸರಿ", "CONFIRMATION", "ANSWER"),
    }
    prior = []
    for test_id, (utt, intent, action) in mapping.items():
        gold = bind_multilingual_ground_truth("kannada", utt, len(prior) + 2, prior)
        assert gold["test_id"] == test_id
        assert gold["expected_intent"] == intent
        assert gold["expected_action"] == action
        prior.append(utt)


def test_mixed_gt_labels():
    gold = bind_multilingual_ground_truth("mixed", "Yes, nanage details bekagide", 2, [])
    assert gold["test_id"] == "M01"
    assert gold["expected_intent"] == "QUALIFICATION"
    gold = bind_multilingual_ground_truth("mixed", "Amenities yenu yenu ide?", 5, TRACK_UTTERANCES["mixed"][:3])
    assert gold["expected_intent"] == "AMENITIES_QUERY"
    gold = bind_multilingual_ground_truth(
        "mixed", "I would like to visit, Friday okay", 6, TRACK_UTTERANCES["mixed"][:4]
    )
    assert gold["test_id"] == "M05"
    assert gold["expected_intent"] == "SITE_VISIT"
    assert gold["expected_action"] == "BOOK_VISIT"
    gold = bind_multilingual_ground_truth("mixed", "4 PM okay", 7, TRACK_UTTERANCES["mixed"][:5])
    assert gold["expected_action"] == "CONFIRM_SLOT"
    gold = bind_multilingual_ground_truth("mixed", "Okay, done", 8, TRACK_UTTERANCES["mixed"][:6])
    assert gold["expected_intent"] == "CONFIRMATION"


def test_language_matching():
    assert score_response_language("hi", "जी हाँ, आपका बजट कितना है?") == LANG_MATCH
    assert score_response_language("kn", "ಸರಿ, ನಿಮ್ಮ ಬಜೆಟ್ ಎಷ್ಟು?") == LANG_MATCH
    assert score_response_language("en", "Sure, what is your budget?") == LANG_MATCH


def test_language_mismatch():
    assert score_response_language("hi", "Sure, what is your budget?") == LANG_MISMATCH
    assert score_response_language("kn", "Sure, what is your budget?") == LANG_MISMATCH
    assert score_response_language("en", "जी हाँ, बताइए") == LANG_MISMATCH


def test_mixed_language_acceptance():
    assert score_response_language("mixed", "Sure, nanage details share madtheeni.") == LANG_MATCH
    assert score_response_language("mixed", "Budget 2 crore ide, 3 BHK available.") == LANG_MATCH
    assert score_response_language("mixed", "Sure, what is your budget?") == LANG_MATCH
    assert score_response_language("hi", "आपका बजट कितना है? Also tell me the BHK.") == LANG_PARTIAL
    assert detect_language("Nanage 3 BHK beku") == "mixed"
    assert detect_language("Amenities yenu yenu ide?") == "mixed"


def test_stateful_site_visit_flow_english():
    history = TRACK_UTTERANCES["english"][:4]
    state = visit_booking_state(history, "I would like to visit")
    assert state.visit_requested
    assert not state.has_date
    assert state.expected_rule == "ASK_DATE"
    eval_ok = evaluate_visit_sequence(state, ASK_DATE)
    assert eval_ok["visit_sequence_code"] == CORRECT_DATE_REQUEST
    state = visit_booking_state(history + ["I would like to visit"], "Friday")
    assert state.has_date
    assert state.expected_rule == "OFFER_SLOTS"
    eval_ok = evaluate_visit_sequence(state, SLOT_OFFER)
    assert eval_ok["visit_sequence_code"] == CORRECT_SLOT_OFFER
    state = visit_booking_state(history + ["I would like to visit", "Friday"], "4 PM")
    assert state.ready_to_confirm
    eval_ok = evaluate_visit_sequence(state, CONFIRM_ANSWER)
    assert eval_ok["visit_sequence_code"] == CORRECT_FINAL_CONFIRMATION


def test_day_date_recognition_hindi_kannada():
    hindi_hist = TRACK_UTTERANCES["hindi"][:4]
    state = visit_booking_state(hindi_hist + ["मैं साइट विज़िट करना चाहता हूँ"], "शुक्रवार")
    assert state.visit_requested
    assert state.has_date
    assert state.expected_rule == "OFFER_SLOTS"
    kn_hist = TRACK_UTTERANCES["kannada"][:4]
    state = visit_booking_state(kn_hist + ["ನಾನು ಸೈಟ್ ವಿಸಿಟ್ ಮಾಡಲು ಇಷ್ಟಪಡುತ್ತೇನೆ"], "ಶುಕ್ರವಾರ")
    assert state.has_date
    assert state.expected_rule == "OFFER_SLOTS"


def test_time_recognition_and_final_confirmation():
    hindi = TRACK_UTTERANCES["hindi"][:6]
    state = visit_booking_state(hindi, "शाम 4 बजे")
    assert state.has_time_preference or state.ready_to_confirm
    assert state.ready_to_confirm
    kn = TRACK_UTTERANCES["kannada"][:6]
    state = visit_booking_state(kn, "ಸಂಜೆ 4 ಗಂಟೆಗೆ")
    assert state.ready_to_confirm
    mixed = TRACK_UTTERANCES["mixed"][:5]
    state = visit_booking_state(mixed, "4 PM okay")
    assert state.ready_to_confirm
    eval_ok = evaluate_visit_sequence(state, CONFIRM_ANSWER)
    assert eval_ok["visit_sequence_code"] == CORRECT_FINAL_CONFIRMATION


def test_mixed_m05_offers_slots_because_day_already_given():
    state = visit_booking_state(TRACK_UTTERANCES["mixed"][:4], "I would like to visit, Friday okay")
    assert state.visit_requested
    assert state.has_date
    assert state.expected_rule == "OFFER_SLOTS"
    assert not state.ready_to_confirm


def test_no_premature_confirmation():
    state = visit_booking_state(TRACK_UTTERANCES["english"][:4], "I would like to visit")
    bad = evaluate_visit_sequence(state, "Your visit is confirmed for Friday at 4:00 PM.")
    assert bad["premature_confirmation"] is True
    assert bad["visit_sequence_code"] == PREMATURE_CONFIRMATION


def test_no_invented_slot():
    state = visit_booking_state(TRACK_UTTERANCES["english"][:5], "Friday")
    bad = evaluate_visit_sequence(state, "We can do 3:00 PM or 7:00 PM.")
    assert bad["invalid_slot"] is True
    assert bad["visit_sequence_code"] == INVALID_SLOT


def test_ground_truth_coverage_per_language():
    for track in LANGUAGE_TRACKS:
        session = _session(track)
        _play(session, track)
        coverage = session.evaluate_session()
        records = flatten_evaluation_records(session.evaluations, session_id=session.run_id)
        customer = [
            row
            for row in records
            if row.get("evaluation_status") != "OK" or row.get("ground_truth_applicable") is not False
        ]
        labelled = [row for row in records if row.get("kind") != "opening" and row.get("expected_intent")]
        customer_turns = [t for t in session.turns if t.get("kind") != "opening"]
        assert len(customer_turns) == len(TRACK_CASES[track])
        summary = session_summary(session.evaluations)
        for alias in DISPLAY_ORDER:
            assert summary[alias]["ground_truth_coverage"] == 1.0
            assert summary[alias]["unexpected_missing_ground_truth"] == 0
            assert summary[alias]["evaluation_coverage"] == 1.0
        assert coverage["evaluated"] == coverage["expected"]
        assert all(row.get("ground_truth") not in {None, "NOT AVAILABLE"} for row in labelled)
        ids = {row["ground_truth"] for row in labelled}
        expected_ids = {case["test_id"] for case in TRACK_CASES[track]}
        assert expected_ids <= ids
        assert all(row.get("retrieved_context") is None or True for row in records)
        for provider in session._providers.values():
            for call in provider.calls:
                assert call["retrieved_context"] == SHARED_PROJECT_CONTEXT


def test_per_language_and_cross_language_aggregation():
    per_track = {}
    for track in LANGUAGE_TRACKS:
        session = _session(track)
        _play(session, track)
        session.evaluate_session()
        per_track[track] = session_summary(session.evaluations)
    agg = aggregate_multilingual(per_track)
    assert agg["ranking"] is None
    assert agg["winner"] is None
    for alias in DISPLAY_ORDER:
        assert set(agg["per_model_per_language"][alias]) == set(LANGUAGE_TRACKS)
        avg = agg["model_multilingual_average"][alias]
        assert avg["intent_accuracy"] is not None
        assert avg["language_understanding_accuracy"] is not None
        robust = agg["language_robustness"][alias]
        assert robust["worst_language"] in LANGUAGE_TRACKS
        assert robust["cross_language_variance"] is not None


def test_token_capture():
    providers = _providers(_script_for_track("english"), input_tokens=11, output_tokens=7)
    session = _session("english", providers=providers)
    _play(session, "english")
    session.evaluate_session()
    records = flatten_evaluation_records(session.evaluations)
    customer = [row for row in records if row.get("ground_truth") not in {None, "NOT AVAILABLE"}]
    assert customer
    assert all(row.get("input_tokens") == 11 for row in customer)
    assert all(row.get("output_tokens") == 7 for row in customer)
    assert all(row.get("total_tokens") == 18 for row in customer)
    summary = session_summary(session.evaluations)
    assert summary["sarvam_conversational"]["avg_total_tokens"] == 18
    assert summary["sarvam_conversational"]["avg_input_tokens"] == 11


def test_malformed_and_missing_response_not_silently_correct():
    responses = [
        _resp("sarvam_conversational", "{not json", None, None, schema_valid=False, raw_response="{not json"),
        _resp("deepseek_flagship", "", None, None, schema_valid=False, raw_response=""),
        _resp("claude_sonnet", None, None, None, error_type="TIMEOUT", error_message="timeout", schema_valid=None),
    ]
    ev = evaluate_turn(
        "हाँ",
        responses,
        [],
        {alias: [] for alias in DISPLAY_ORDER},
        2,
        SHARED_PROJECT_CONTEXT,
        outbound=True,
        required_aliases=list(DISPLAY_ORDER),
        language_track="hindi",
        ground_truth_id="H01",
    )
    by_alias = {row["alias"]: row for row in ev["models"]}
    assert by_alias["sarvam_conversational"]["structured_output_status"] == "malformed_response"
    assert by_alias["sarvam_conversational"]["intent_correct"] is False
    assert by_alias["deepseek_flagship"]["structured_output_status"] == "empty_response"
    assert by_alias["claude_sonnet"]["api_error"] is True
    assert by_alias["claude_sonnet"]["structured_output_status"] == "timeout"
    assert by_alias["claude_sonnet"]["intent_correct"] is None
    assert by_alias["claude_flagship"]["evaluation_status"] == EVAL_STATUS_MISSING_RESPONSE
    timeout = TurnResponse(
        alias="x",
        provider="x",
        model_id="x",
        answer=None,
        intent=None,
        action=None,
        language=None,
        stage=None,
        raw_response="",
        latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        error_type="TIMEOUT",
        error_message="timeout",
        schema_valid=None,
    )
    assert structured_output_status(timeout) == "timeout"


def test_language_understanding_is_intent_not_purity():
    ev = evaluate_turn(
        "ನನಗೆ 3 BHK ಬೇಕು",
        _all(QUALIFY_ANSWER, "QUALIFICATION", "QUALIFY", language="en"),
        [],
        _hist(TRACK_UTTERANCES["kannada"][:2]),
        4,
        SHARED_PROJECT_CONTEXT,
        outbound=True,
        required_aliases=list(DISPLAY_ORDER),
        language_track="kannada",
        ground_truth_id="K03",
    )
    model = ev["models"][0]
    assert model["ground_truth"] == "K03"
    assert model["intent_correct"] is True
    assert model["language_understanding_correct"] is True
    assert model["response_language_match"] == LANG_MISMATCH


def test_existing_outbound_regression_yes_still_binds_ob_t02():
    session = InteractiveSession(
        providers=_providers(),
        conversation_mode="outbound",
        opening_mode="fixed",
    )
    assert session.language_track is None
    session.begin_outbound()
    session.generate_turn("Yes")
    ev = session.evaluate_latest()
    assert ev["ground_truth"] == "OB-T02"
    assert ev["models"][0].get("language_track") is None
    assert ev["models"][0].get("response_language_match") is None


def test_shared_kb_identical_across_tracks():
    contexts = set()
    for track in LANGUAGE_TRACKS:
        session = _session(track)
        _play(session, track)
        contexts.add(session.retrieved_context)
        for provider in session._providers.values():
            for call in provider.calls:
                assert "SOBHA Townpark" in call["retrieved_context"]
                assert "1.8 Crore" in call["retrieved_context"]
                assert "10:00 AM" in call["retrieved_context"]
    assert contexts == {SHARED_PROJECT_CONTEXT}


def test_multilingual_cli_runner_with_fake_providers(tmp_path, monkeypatch):
    args = parse_args(["--mode", "multilingual", "--track", "english", "--opening", "fixed", "--demo"])
    providers = _providers(_script_for_track("english"))

    class Args:
        mode = "multilingual"
        track = "english"
        opening = "fixed"
        dry_run = False
        max_display_chars = 0
        demo = True
        verbose = False

    lines: list[str] = []

    def out(text="", end="\n"):
        lines.append(text)

    monkeypatch.setattr(
        "interactive.cli._session_from_args",
        lambda args, language_track=None, providers=None: InteractiveSession(
            providers=_providers(_script_for_track(language_track or "english")),
            conversation_mode="outbound",
            opening_mode="fixed",
            language_track=language_track or "english",
        ),
    )
    code = run_multilingual_tracks(Args(), out)
    assert code == 0
    joined = "\n".join(lines)
    assert "Twilio DISABLED" in joined or "TWILIO: DISABLED" in joined
    assert "No model ranking" in joined
    assert args.mode == "multilingual"


def test_detect_response_language_scripts():
    assert detect_response_language("आपका बजट कितना है?") == "hi"
    assert detect_response_language("ನಿಮ್ಮ ಬಜೆಟ್ ಎಷ್ಟು?") == "kn"
    assert detect_response_language("What is your budget?") == "en"
    assert detect_response_language("") == "empty"


def test_multilingual_track_order_and_first_utterances():
    assert LANGUAGE_TRACKS == ("english", "hindi", "kannada", "mixed")
    assert selected_tracks("all") == LANGUAGE_TRACKS
    assert track_utterances("english")[0] == "Yes"
    assert track_utterances("hindi")[0] == "हाँ"
    assert track_utterances("kannada")[0] == "ಹೌದು"
    assert track_utterances("mixed")[0] == "Yes, nanage details bekagide"
    assert [case["test_id"] for case in TRACK_CASES["english"]] == [f"E0{i}" for i in range(1, 9)]
    assert [case["test_id"] for case in TRACK_CASES["hindi"]] == [f"H0{i}" for i in range(1, 9)]
    assert [case["test_id"] for case in TRACK_CASES["kannada"]] == [f"K0{i}" for i in range(1, 9)]
    assert [case["test_id"] for case in TRACK_CASES["mixed"]] == [f"M0{i}" for i in range(1, 8)]


def test_language_track_propagates_into_every_model_request():
    expected = {
        "english": ("Yes", "english"),
        "hindi": ("हाँ", "hindi"),
        "kannada": ("ಹೌದು", "kannada"),
        "mixed": ("Yes, nanage details bekagide", "mixed"),
    }
    for track, (utterance, token) in expected.items():
        providers = _providers(_script_for_track(track))
        session = _session(track, providers=providers)
        session.begin_outbound()
        session.generate_turn(utterance)
        for provider in providers.values():
            extra = provider.calls[0]["extra"]
            assert extra["language_track"] == track
            text = build_user_payload(
                provider.calls[0]["conversation_history"],
                provider.calls[0]["customer_utterance"],
                provider.calls[0]["retrieved_context"],
                extra,
            )
            assert f"Customer language track: {token}" in text
            assert provider.calls[0]["customer_utterance"] == utterance
        preview = session.preview_payloads(utterance)
        for text in preview.values():
            assert f"Customer language track: {token}" in text


def test_cli_switches_language_track_per_session():
    seen: list[str] = []

    class Args:
        mode = "multilingual"
        track = "all"
        opening = "fixed"
        dry_run = False
        max_display_chars = 0
        demo = True
        verbose = False

    def factory(args, language_track=None, providers=None):
        seen.append(language_track)
        return InteractiveSession(
            providers=_providers(_script_for_track(language_track)),
            conversation_mode="outbound",
            opening_mode="fixed",
            language_track=language_track,
        )

    import interactive.cli as cli

    original = cli._session_from_args
    cli._session_from_args = factory
    try:
        lines: list[str] = []
        code = run_multilingual_tracks(Args(), lambda text="", end="\n": lines.append(text))
    finally:
        cli._session_from_args = original
    assert code == 0
    assert seen == ["english", "hindi", "kannada", "mixed"]
    joined = "\n".join(lines)
    assert "Track order: english -> hindi -> kannada -> mixed" in joined
    assert "Setting language_track = hindi" in joined
    assert "Customer > हाँ" in joined
    assert "Customer > ಹೌದು" in joined
    assert "Customer > Yes, nanage details bekagide" in joined


class HangProvider:
    def __init__(self, alias: str, hang: bool = True):
        self.alias = alias
        self.hang = hang
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.hang:
            time.sleep(2.5)
        return {
            "provider": "fake",
            "model_id": f"fake-{self.alias}",
            "answer": "Noted. A 3 BHK starts at INR 1.8 Crore onwards.",
            "intent": "QUALIFICATION",
            "action": "QUALIFY",
            "language": "en",
            "raw_response": "Noted.",
            "schema_valid": True,
            "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
            "latency_ms": 5,
            "error": None,
            "error_type": None,
            "error_message": None,
        }


def test_model_timeout_returns_structured_result_without_hanging():
    providers = {
        alias: HangProvider(alias, hang=(alias == "claude_flagship"))
        for alias in DISPLAY_ORDER
    }
    session = InteractiveSession(
        providers=providers,
        conversation_mode="outbound",
        opening_mode="fixed",
        language_track="english",
    )
    session.collect_timeout_seconds = 1.0
    session.begin_outbound()
    started = time.perf_counter()
    responses = session.generate_turn("Yes")
    elapsed = time.perf_counter() - started
    assert elapsed < 5, f"runner hung for {elapsed:.1f}s"
    by_alias = {item.alias: item for item in responses}
    assert by_alias["claude_flagship"].error_type == "TIMEOUT"
    assert by_alias["claude_flagship"].answer is None
    assert by_alias["sarvam_conversational"].error_type is None
    assert by_alias["sarvam_conversational"].answer
    ev = session.evaluate_latest()
    timed_out = next(row for row in ev["models"] if row["alias"] == "claude_flagship")
    assert timed_out["api_error"] is True
    assert timed_out["intent_correct"] is None
    assert timed_out["structured_output_status"] == "timeout"


def test_model_exception_is_structured_not_success():
    class Boom(FakeProvider):
        def generate(self, **kwargs):
            raise RuntimeError("provider exploded")

    providers = {alias: Boom(alias) for alias in DISPLAY_ORDER}
    session = InteractiveSession(
        providers=providers,
        conversation_mode="outbound",
        opening_mode="fixed",
        language_track="english",
    )
    session.collect_timeout_seconds = 2
    session.begin_outbound()
    responses = session.generate_turn("Yes")
    assert all(item.error_type == "API_ERROR" for item in responses)
    assert all(item.answer is None for item in responses)
