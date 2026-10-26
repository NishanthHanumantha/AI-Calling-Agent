# Version baseline

Version: V10 frozen / V11 development

`Version11/` is the development copy. `Version10/`, `Version9/`, `Version8/`, `Version7/`, `Version6/`, `Version5/`, `Version4/`, and `Version3/` stay frozen.

Copied from `Version10/` on 26 October 2026, and only what the localhost call needs:

- `app_v3.py` — the all-Sobha Bengaluru conversation
- `sobha-townpark-brochure.pdf` — the Townpark facts that came with that copy
- `local_call/` — the microphone page, Sarvam speech-to-text, Bulbul, clause-by-clause speech, and the Opus 5.5 scorecard
- `tests/test_human_like_v1.py` and `tests/test_local_call_evaluator.py`
- `TTS_RESEARCH.md` — the ElevenLabs and Cartesia notes
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: saved call runs, the comparison reports, `LLM_Evaluation/`, and `make_call.py`. The shared brochure folder stays at the repository root.

This copy still speaks the English Version 10 call. A request for Hindi, Kannada, Tamil, Telugu, or Malayalam is still answered with “I can continue in English.” The next development keeps English as the language of the call until the customer asks for another one, and then adds Hindi and Kannada first, followed by Tamil, Telugu, and Malayalam. The caller, Ishita, pace, and temperature stay as they are.

## Speech on this page

Caller: Sarvam `sarvam-105b-conversations`. Speech recognition: Sarvam `saaras:v4`.

Spoken reply: Sarvam Bulbul v3, speaker `ishita`. Where a reply is written as `sq. ft.`, Ishita is given the words “square feet.” If Bulbul returns no audio, Neerja speaks that line.

The phone path is not in this folder. Polly Kajal-Neural is not used here.

## Port

`http://127.0.0.1:8018/` when this copy is started.

Version 10 stays on `http://127.0.0.1:8017/`. Version 9 stays on `http://127.0.0.1:8016/`. Version 8 stays on `http://127.0.0.1:8015/`. Version 7 stays on `http://127.0.0.1:8014/`. Version 6 stays on `http://127.0.0.1:8013/`. Version 5 stays on `http://127.0.0.1:8012/`. Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
