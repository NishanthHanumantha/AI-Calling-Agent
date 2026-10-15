# Version baseline

Version: V7 frozen / V8 development

`Version8/` is the development copy. `Version7/`, `Version6/`, `Version5/`, `Version4/`, and `Version3/` stay frozen.

Copied from `Version7/` on 15 October 2026, and only what the localhost call needs:

- `app_v3.py` — the Townpark conversation, including an open answer, one visit offer, and a polite close
- `sobha-townpark-brochure.pdf` — the project facts
- `local_call/` — the microphone page, Sarvam speech-to-text, Bulbul, clause-by-clause speech, and the Opus 5.5 scorecard
- `tests/test_human_like_v1.py` and `tests/test_local_call_evaluator.py`
- `TTS_RESEARCH.md` — the ElevenLabs and Cartesia notes
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: saved call runs, the comparison reports, `LLM_Evaluation/`, and `make_call.py`.

The Version 7 behavior comes with this copy. Speech starts on the first clause. “sq. ft.” is spoken as “square feet.” An undecided topic stays open. A no to one topic can still hear the visit once. A no to the project, a deferred visit, or a request for no details closes with one line. A short noise does not stop Ishita.

## Speech on this page

Caller: Sarvam `sarvam-105b-conversations`. Speech recognition: Sarvam `saaras:v4`.

Spoken reply: Sarvam Bulbul v3, speaker `ishita`. Where a reply is written as `sq. ft.`, Ishita is given the words “square feet.” If Bulbul returns no audio, Neerja speaks that line.

The phone path is not in this folder. Polly Kajal-Neural is not used here.

## Port

`http://127.0.0.1:8015/` when this copy is started.

Version 7 stays on `http://127.0.0.1:8014/`. Version 6 stays on `http://127.0.0.1:8013/`. Version 5 stays on `http://127.0.0.1:8012/`. Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
