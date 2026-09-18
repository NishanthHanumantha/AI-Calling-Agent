# LLM Evaluation 1.1.0 — Sarvam + Claude

Offline / programmatic benchmark for the **LLM layer** of the AI Calling Agent.

This module does **not** make Twilio calls, does **not** use live audio, STT, or TTS, and does **not** change production `app_v3.py`.

## 1. Purpose

Compare **two Sarvam models** and **two Claude models** on the same 10 golden cases (up to 40 executions). Metrics are multi-dimensional. There is **no winner / rank column**.

LLM-EVAL.1 baseline (`output/runs/20260918_071117/`) is preserved and is **not** merged into 1.1 scores.

## 2. Why two models per provider

- **Conversational / Sonnet:** closer to voice-agent latency and spoken style.
- **Flagship / Opus:** higher-intelligence / reasoning variant.

This isolates provider family vs model-tier differences.

## 3. Providers in scope

**In scope:** Sarvam, Claude (Anthropic).

**Out of scope:** Qwen. The Qwen adapter may remain in the repo from LLM-EVAL.1 but is **disabled**. Missing `QWEN_API_KEY` does not fail this phase.

## 4. Model roles

| Alias | Provider | Role | Default model ID (overridable) |
|---|---|---|---|
| `sarvam_conversational` | sarvam | conversational | `sarvam-105b-conversations` (same as production V3) |
| `sarvam_flagship` | sarvam | flagship | `sarvam-105b` |
| `claude_flagship` | claude | flagship | `claude-opus-4-8` |
| `claude_sonnet` | claude | general_purpose | `claude-sonnet-4-6` |

IDs come from environment variables, then yaml defaults. **No silent substitution** if an ID is unavailable.

## 5. Model configuration

Set in repo `.env` (gitignored) or `LLM_Evaluation/.env`:

```
SARVAM_API_KEY=
SARVAM_CONVERSATIONAL_MODEL=sarvam-105b-conversations
SARVAM_FLAGSHIP_MODEL=sarvam-105b
ANTHROPIC_API_KEY=
CLAUDE_FLAGSHIP_MODEL=claude-opus-4-8
CLAUDE_SONNET_MODEL=claude-sonnet-4-6
```

See `.env.example`. Do not commit secrets.

## 6. Claude model availability

Previous LLM-EVAL.1 Claude 404s were **model ID not found**, not a missing API key.

`python run_evaluation.py --check-models`:

- Uses existing `ANTHROPIC_API_KEY`
- Lists Anthropic models (`GET /v1/models`)
- Distinguishes `AUTHENTICATION = PASS` vs `MODEL = NOT FOUND`
- Classifies: `AVAILABLE`, `MODEL_NOT_FOUND`, `AUTH_ERROR`, `PERMISSION_ERROR`, `RATE_LIMIT`, `OTHER_API_ERROR`

404 is not retried. 429 / timeout / 5xx retry up to twice.

## 7. `--check-models`

```bash
cd LLM_Evaluation
python run_evaluation.py --check-models
```

Prints Sarvam conversational/flagship and Claude flagship/sonnet. Qwen: OUT OF SCOPE — NOT TESTED.

## 8. `--dry-run`

```bash
python run_evaluation.py --dry-run
```

Zero API calls. Reports 4 configured models (2 Sarvam, 2 Claude), 10 cases, 40 potential executions.

## 9. Benchmark

```bash
python run_evaluation.py --models all
python run_evaluation.py --models sarvam_conversational,sarvam_flagship
python run_evaluation.py --provider claude
```

Unavailable models are skipped. API failures are not scored as 0% quality.

## 10. Evaluation metrics

Intent, action, facts, grounding, hallucination, context, language, schema, 1–5 response quality (relevance, clarity, completeness, conversational + **stage appropriateness**), latency, tokens.

Text-only extras (not audio): word count, conciseness, actionability, **TEXT-LEVEL SHORT-UTTERANCE SIMULATION**.

Cost: `NOT_CONFIGURED` unless pricing is added later.

LLM-as-a-judge is optional (`NOT_RUN` if unset).

## 11. Excel output

Each run: `output/runs/<run_id>/`

- `raw_outputs.jsonl`
- `evaluation_results.csv`
- `evaluation_summary.xlsx`
- `run_metadata.json`

Sheets: Executive Summary, Provider Comparisons (Sarvam conversational vs flagship, Claude flagship vs sonnet), Detailed Results, Model × Test Matrix, Error Analysis, Multilingual (smoke test only), Latency & Usage, Model Configuration.

## 12. Error classification

API: `MODEL_NOT_FOUND`, `AUTH_ERROR`, `PERMISSION_ERROR`, `RATE_LIMIT`, `TIMEOUT`, `API_ERROR`.

Quality (successful completions only): `WRONG_INTENT`, `WRONG_ACTION`, `MISSING_FACT`, `HALLUCINATION`, `CONTEXT_FAILURE`, `LANGUAGE_FAILURE`, `STAGE_FAILURE`, `POOR_RESPONSE_QUALITY`, `INVALID_JSON`.

## 13. Add another model later

1. Adapter implementing `LLMProvider.generate`
2. Register in `models/PROVIDER_CLASSES`
3. Add an enabled alias in `config/models.yaml`

Do not add Qwen to `active_model_aliases` until that phase.

## 14. Why Qwen is out of scope

No `QWEN_API_KEY` in this environment. LLM-EVAL.1.1 uses only Sarvam + Claude.

## 15. Offline text vs real voice

This is **offline LLM evaluation**. It does not measure STT errors, barge-in, TTS, or Twilio audio. Short utterances are **text-level simulations** only.
