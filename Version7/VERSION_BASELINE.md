# Version baseline

Version: V7 frozen / V8 development

`Version7/` is frozen as of 15 October 2026. It keeps the localhost architecture already in this folder: Sarvam as the caller, Sarvam speech-to-text, Sarvam Bulbul v3 speaker `ishita`, “sq. ft.” spoken as “square feet,” and speech that starts on the first clause. Neerja speaks only when Bulbul returns no audio. Do not modify `Version7/`, `Version6/`, `Version5/`, `Version4/`, or `Version3/`. Further development belongs in `Version8/`.

The notes below record how Version 7 was built. They are historical.

Version at the time of those notes: V6 frozen / V7 development

`Version7/` was the development copy. `Version6/`, `Version5/`, `Version4/`, and `Version3/` stay frozen.

Copied from `Version6/` on 13 October 2026, and only what the localhost call needs:

- `app_v3.py` — the same Townpark conversation
- `sobha-townpark-brochure.pdf` — the project facts
- `local_call/` — the microphone page, Sarvam speech-to-text, Bulbul, and the Opus 5.5 scorecard
- `tests/test_human_like_v1.py` and `tests/test_local_call_evaluator.py`
- `TTS_RESEARCH.md` — the ElevenLabs and Cartesia notes
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: saved call runs, the architecture Word report, `LLM_Evaluation/`, and `make_call.py`.

## Speech on this page

Caller: Sarvam `sarvam-105b-conversations`. Speech recognition: Sarvam `saaras:v4`.

Spoken reply: Sarvam Bulbul v3, speaker `ishita`. Where a reply is written as `sq. ft.`, Ishita is given the words “square feet.” If Bulbul returns no audio, Neerja speaks that line.

The phone path is not in this folder. Polly Kajal-Neural is not used here.

## Port

`http://127.0.0.1:8014/` when this copy is started.

Version 6 stays on `http://127.0.0.1:8013/`. Version 5 stays on `http://127.0.0.1:8012/`. Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
