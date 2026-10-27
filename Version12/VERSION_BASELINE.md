# Version baseline

Version: V11 frozen / V12 development

`Version12/` is the development copy. `Version11/`, `Version10/`, `Version9/`, `Version8/`, `Version7/`, `Version6/`, `Version5/`, `Version4/`, and `Version3/` stay frozen.

Copied from `Version11/` on 27 October 2026, and only what the localhost call needs:

- `app_v3.py` — the all-Sobha Bengaluru conversation, including Hindi, Kannada, Tamil, Telugu, and Malayalam
- `local_call/` — the microphone page, Sarvam speech-to-text, Bulbul, clause-by-clause speech, and the Opus 5.5 scorecard
- `tests/test_human_like_v1.py` and `tests/test_local_call_evaluator.py`
- `TTS_RESEARCH.md` — the ElevenLabs and Cartesia notes
- `outbound_customer_calls_e7357390.plan.md` — the dialer to build here
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: saved call runs, the Townpark brochure that had been left inside Version 11, and `make_call.py`. The shared brochure folder stays at the repository root.

This copy speaks the frozen Version 11 call. Outbound calls to customers are the next work, in this folder only. The caller, Ishita, pace, and temperature stay as they are.

## Speech on this page

Caller: Sarvam `sarvam-105b-conversations`. Speech recognition: Sarvam `saaras:v4`.

Spoken reply: Sarvam Bulbul v3, speaker `ishita`. Where a reply is written as `sq. ft.`, Ishita is given the words “square feet.” If Bulbul returns no audio, Neerja speaks that line in English. Hindi, Kannada, Tamil, Telugu, and Malayalam stay for the rest of the call once the customer uses them.

The phone path is not in this folder yet. Polly Kajal-Neural is not used on the local page.

## Port

`http://127.0.0.1:8019/` when this copy is started.

Version 11 stays on `http://127.0.0.1:8018/`. Version 10 stays on `http://127.0.0.1:8017/`. Version 9 stays on `http://127.0.0.1:8016/`. Version 8 stays on `http://127.0.0.1:8015/`. Version 7 stays on `http://127.0.0.1:8014/`. Version 6 stays on `http://127.0.0.1:8013/`. Version 5 stays on `http://127.0.0.1:8012/`. Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
