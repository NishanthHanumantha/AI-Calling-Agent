# Version baseline

Version: V11 frozen / V13 development

`Version13/` is the development copy. `Version11/`, `Version12/`, `Version10/`, `Version9/`, `Version8/`, `Version7/`, `Version6/`, `Version5/`, `Version4/`, and `Version3/` stay frozen.

Copied from `Version11/` on 5 November 2026, and only what the localhost call needs:

- `app_v3.py` — the all-Sobha Bengaluru conversation
- `local_call/` — the microphone page, Sarvam speech-to-text, Bulbul, clause-by-clause speech, translation, and the scorecard
- `tests/` — the local-call, usage, model, phone, and human-likeness tests
- `TTS_RESEARCH.md` — the ElevenLabs and Cartesia notes
- `outbound_customer_calls_e7357390.plan.md` — the dialer notes that came with Version 11
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: saved call runs, cache files, and `sobha-townpark-brochure.pdf`. The shared brochure folder stays at the repository root.

This copy starts from the Version 11 call of 5 November 2026. A language the customer names, including English, is spoken on that same turn and stays until they name another language. The caller, Ishita, pace, and temperature stay as they are.

## Speech on this page

Caller: Sarvam `sarvam-105b-conversations`. Speech recognition: Sarvam `saaras:v4`.

Spoken reply: Sarvam Bulbul v4 Flash, speaker `ishita_enhi_customer`. Where a reply is written as `sq. ft.`, Ishita is given the words "square feet." If Bulbul returns no audio, Neerja speaks that line. Hindi, Kannada, Tamil, Telugu, and Malayalam are translated before Bulbul speaks them.

The phone path is not the local page. Polly is not used on the local page.

## Port

`http://127.0.0.1:8020/` when this copy is started.

Version 12 stays on `http://127.0.0.1:8019/`. Version 11 stays on `http://127.0.0.1:8018/`. Version 10 stays on `http://127.0.0.1:8017/`. Version 9 stays on `http://127.0.0.1:8016/`. Version 8 stays on `http://127.0.0.1:8015/`. Version 7 stays on `http://127.0.0.1:8014/`. Version 6 stays on `http://127.0.0.1:8013/`. Version 5 stays on `http://127.0.0.1:8012/`. Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
