# Version baseline

Version: V6 frozen / V7 development

`Version6/` is frozen as of 13 October 2026. It keeps the localhost architecture already in this folder: Sarvam as the caller, Sarvam speech-to-text, Sarvam Bulbul v3 speaker `ishita` for the spoken reply, “sq. ft.” spoken as “square feet,” and Neerja only when Bulbul returns no audio. Do not modify `Version6/`, `Version5/`, `Version4/`, or `Version3/`. Further development belongs in `Version7/`.

The notes below record how Version 6 was built. They are historical.

Version at the time of those notes: V5 frozen / V6 development

`Version6/` was the development copy. `Version5/`, `Version4/`, and `Version3/` stay frozen.

Copied from `Version5/` on 12 October 2026, and only what the localhost call needs:

- `app_v3.py` — the same Townpark conversation
- `sobha-townpark-brochure.pdf` — the project facts
- `local_call/` — the microphone page, Sarvam speech-to-text, Bulbul, and the Opus 5.5 scorecard
- `tests/test_human_like_v1.py` and `tests/test_local_call_evaluator.py`
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: `LLM_Evaluation/`, `make_call.py`, benchmark audio, and saved call runs.

## Speech on this page

The copied page still speaks with Sarvam Bulbul v3, speaker `ishita`. Neerja speaks only when Bulbul returns no audio. ElevenLabs and Cartesia are not wired. The research is in `TTS_RESEARCH.md`.

Caller: Sarvam `sarvam-105b-conversations`. Speech recognition: Sarvam `saaras:v4`.

The phone path is not in this folder. Polly Kajal-Neural is not used here.

## Port

`http://127.0.0.1:8013/` when this copy is started.

Version 5 stays on `http://127.0.0.1:8012/`. Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
