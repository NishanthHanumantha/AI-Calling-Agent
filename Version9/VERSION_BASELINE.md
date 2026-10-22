# Version baseline

Version: V9 frozen / V10 development

`Version9/` is frozen as of 22 October 2026. It is the SOBHA Townpark call: Sarvam as the caller, Sarvam speech-to-text, Sarvam Bulbul v3 speaker `ishita`, “sq. ft.” spoken as “square feet,” speech that starts on the first clause, and a fixed line for an unmatched positive, negative, neutral, or ambiguous reply before the model is called. Neerja speaks only when Bulbul returns no audio. Do not modify `Version9/`, `Version8/`, `Version7/`, `Version6/`, `Version5/`, `Version4/`, or `Version3/`. Further development belongs in `Version10/`. Versions through Version 9 stay Townpark. All Sobha projects belong in Version 10 and above.

The notes below record how Version 9 was built. They are historical.

Version at the time of those notes: V8 frozen / V9 development

`Version9/` was the development copy. `Version8/`, `Version7/`, `Version6/`, `Version5/`, `Version4/`, and `Version3/` stay frozen.

Copied from `Version8/` on 21 October 2026, and only what the localhost call needs:

- `app_v3.py` — the Townpark conversation
- `sobha-townpark-brochure.pdf` — the project facts
- `local_call/` — the microphone page, Sarvam speech-to-text, Bulbul, clause-by-clause speech, and the Opus 5.5 scorecard
- `tests/test_human_like_v1.py` and `tests/test_local_call_evaluator.py`
- `TTS_RESEARCH.md` — the ElevenLabs and Cartesia notes
- `requirements.txt` and `.gitignore`
- `.env` is a local copy and is not recorded here

Not copied: saved call runs, the comparison reports, `LLM_Evaluation/`, and `make_call.py`.

The Version 8 behavior comes with this copy. Speech starts on the first clause. “sq. ft.” is spoken as “square feet.” An undecided topic stays open. A no to one topic can still hear the visit once. A yes to a visit closes with the sales-advisor line. A short noise does not stop Ishita.

## Speech on this page

Caller: Sarvam `sarvam-105b-conversations`. Speech recognition: Sarvam `saaras:v4`.

Spoken reply: Sarvam Bulbul v3, speaker `ishita`. Where a reply is written as `sq. ft.`, Ishita is given the words “square feet.” If Bulbul returns no audio, Neerja speaks that line.

The phone path is not in this folder. Polly Kajal-Neural is not used here.

## Port

`http://127.0.0.1:8016/` when this copy is started.

Version 8 stays on `http://127.0.0.1:8015/`. Version 7 stays on `http://127.0.0.1:8014/`. Version 6 stays on `http://127.0.0.1:8013/`. Version 5 stays on `http://127.0.0.1:8012/`. Version 4 stays on `http://127.0.0.1:8010/`. Version 3 stays on `http://127.0.0.1:8011/`.
