# Version baseline

Version: V10 frozen / V11 development

`Version10/` is frozen as of 26 October 2026. It is the all-Sobha Bengaluru call: Sarvam as the caller, Sarvam speech-to-text, Sarvam Bulbul v3 speaker `ishita`, “sq. ft.” spoken as “square feet,” speech that starts on the first clause, and a shortlist that stays in the region the customer named. The call stays in English. Neerja speaks only when Bulbul returns no audio. Do not modify `Version10/`, `Version9/`, `Version8/`, `Version7/`, `Version6/`, `Version5/`, `Version4/`, or `Version3/`. Further development belongs in `Version11/`. Hindi, Kannada, and the other languages belong in Version 11.

The notes below record how Version 10 was built. They are historical.

Version at the time of those notes: V9 frozen / V10 development

`Version10/` was the development copy. `Version9/`, `Version8/`, `Version7/`, `Version6/`, `Version5/`, `Version4/`, and `Version3/` stay frozen.

Copied from `Version9/` on 22 October 2026, and only what the localhost call needs:

- `app_v3.py` — the conversation copied from Version 9
- `sobha-townpark-brochure.pdf` — the Townpark facts that came with that copy
- `local_call/` — the microphone page, Sarvam speech-to-text, Bulbul, clause-by-clause speech, and the Opus 5.5 scorecard
- `tests/test_human_like_v1.py` and `tests/test_local_call_evaluator.py`
- `TTS_RESEARCH.md` — the ElevenLabs and Cartesia notes
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: saved call runs, the comparison reports, `LLM_Evaluation/`, and `make_call.py`.

Through Version 9 the call is for SOBHA Townpark only. From Version 10 onward the work is for all Sobha projects. This copy still speaks the Townpark conversation it was copied with, including a fixed line for an unmatched positive, negative, neutral, or ambiguous reply. The next development replaces that single-project call with a call that can cover the Sobha projects.

## Speech on this page

Caller: Sarvam `sarvam-105b-conversations`. Speech recognition: Sarvam `saaras:v4`.

Spoken reply: Sarvam Bulbul v3, speaker `ishita`. Where a reply is written as `sq. ft.`, Ishita is given the words “square feet.” If Bulbul returns no audio, Neerja speaks that line.

The phone path is not in this folder. Polly Kajal-Neural is not used here.

## Port

`http://127.0.0.1:8017/` when this copy is started.

Version 9 stays on `http://127.0.0.1:8016/`. Version 8 stays on `http://127.0.0.1:8015/`. Version 7 stays on `http://127.0.0.1:8014/`. Version 6 stays on `http://127.0.0.1:8013/`. Version 5 stays on `http://127.0.0.1:8012/`. Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
