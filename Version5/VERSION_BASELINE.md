# Version baseline

Version: V5 frozen / V6 development

`Version5/` is frozen as of 12 October 2026. It keeps the localhost architecture already in this folder: Sarvam as the caller, Sarvam speech-to-text, Sarvam Bulbul v3 speaker `ishita` for the spoken reply, and Neerja only when Bulbul returns no audio. Do not modify `Version5/`, `Version4/`, or `Version3/`. Research on ElevenLabs and Cartesia belongs in `Version6/`.

The notes below record how Version 5 was built. They are historical.

Version at the time of those notes: V4 frozen / V5 development

`Version5/` was the development copy for the localhost speech trial. `Version4/` and `Version3/` stay frozen.

Copied from `Version4/` on 12 October 2026, and only what the localhost call needs:

- `app_v3.py` — the same Townpark conversation
- `sobha-townpark-brochure.pdf` — the project facts
- `local_call/` — the microphone page, Sarvam speech-to-text, and the Opus 5.5 scorecard
- `tests/test_human_like_v1.py` and `tests/test_local_call_evaluator.py`
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: `LLM_Evaluation/`, `make_call.py`, benchmark audio, and saved call runs.

## Speech on this page

Caller: Sarvam `sarvam-105b-conversations`. Unchanged from Version 4.

Speech recognition: Sarvam `saaras:v4`. Unchanged from Version 4.

Spoken reply: Sarvam Bulbul v3, speaker `ishita`, English (`en-IN`), pace 1.0, temperature 0.6. If Bulbul returns no audio, the page speaks with Neerja.

The phone path is not in this folder. Polly Kajal-Neural is not used here.

## Port

`http://127.0.0.1:8012/`

Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
