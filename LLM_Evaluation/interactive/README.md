# Interactive Multi-Model Conversation Lab (LLM-EVAL.2)

Local-only evaluation console for the AI Calling Agent. Type one customer message and compare four configured LLMs side by side.

This is **not** a phone call. Twilio is never invoked. Production `app_v3.py`, `/voice`, and STT/TTS are not used.

## 1. Purpose

Interactively compare conversation intelligence across:

1. Sarvam — Conversational (`sarvam_conversational`)
2. Sarvam — Flagship (`sarvam_flagship`)
3. Claude — Sonnet (`claude_sonnet`)
4. Claude — Flagship (`claude_flagship`)

Dimensions: intent, action, context retention, stage handling, factual correctness, grounding, hallucination, response quality, multilingual/code-switching.

Qwen is out of scope.

## 2. Architecture

```
                  USER
                   |
                   v
          INTERACTIVE CLI
                   |
          Common conversation context
          (same user text, same KB, same system role)
                   |
     +--------+--------+--------+
     v        v        v        v
 Sarvam C  Sarvam F  Claude S  Claude F
 (isolated histories — a model never sees another model's replies)
                   |
          RESPONSE COLLECTION
                   |
        DISPLAY / EVALUATE / EXPORT
```

Adapters and `config/models.yaml` are reused from LLM-EVAL.1.1. No second `.env`. Keys come from the existing project `.env`.

Model-specific API formatting (Sarvam chat-completions vs Anthropic messages) is handled by existing providers. The **semantic** payload is identical: same system prompt, same retrieved Townpark KB, same customer utterance, same stage hint. Each model additionally receives **its own** prior assistant turns so follow-ups such as "Morning would be better" stay fair without leaking another model's wording.

## 3. How to launch

From the repository root:

```bash
python -m LLM_Evaluation.interactive.cli
```

Equivalent:

```bash
python -m interactive.cli
```

(run from the `LLM_Evaluation/` directory)

Startup banner:

```
============================================================
          AI CALLING AGENT
      INTERACTIVE LLM COMPARISON LAB
============================================================

Mode:
LOCAL INTERACTIVE EVALUATION

Twilio:
DISABLED
```

Then:

```
Customer >
```

## 4. Conversation history

Each model keeps an independent history:

```
[user] Hi, I am interested in Sobha Townpark.
[assistant] <that model's reply only>
[user] Where is it located?
```

Turn 2 includes Turn 1 **for that model only**. `/reset` clears all four histories and restarts at TURN 01.

## 5. Four-model comparison

Every customer line is sent to all four models in parallel. One provider error does not abort the others. Failed models show `STATUS: ERROR` with a redacted message. API errors are **not** scored as hallucination or poor quality.

## 6. Commands

| Command | Meaning |
| --- | --- |
| `/help` | Show commands |
| `/reset` | New conversation |
| `/history` | Customer turns only (no API calls) |
| `/evaluate` | Evaluate the latest turn |
| `/evaluate session` | Aggregate metrics for evaluated turns |
| `/save` / `/export` | Write JSON/JSONL/CSV/XLSX |
| `/models` | Configured IDs and availability (no keys) |
| `/status` | Last request status and latency |
| `/debug` | Latest user message, histories, KB, request metadata |
| `/quit` | Exit |

Example:

```
Customer > /help
    /help      Show available commands
    /reset     Start a new conversation
    ...
```

## 7. Evaluation metrics

`/evaluate` never names a winner.

When a golden-dataset case matches the customer utterance:

- Intent / action: Accuracy, Precision, Recall, Macro F1 across labeled turns
- Stage correctness from expected action / inferred calling-agent stage
- Language correctness when the golden row specifies language

Always (no invented labels):

- Grounding against `PROJECT_KB` (retrieved context)
- Unsupported claim count / hallucination (KB numbers/phrases, not style differences)
- Context retention for multi-turn follow-ups
- Quality 1–5: relevance, completeness, clarity, conversational appropriateness, stage appropriateness
- Latency and provider token usage when present (`N/A` if the API omitted them — never estimated)

If no golden row matches: `Ground Truth: NOT AVAILABLE`. Intent/action correctness are left blank.

## 8. Grounding methodology

All four models receive the same Townpark `PROJECT_KB` snapshot (not live PDF chunks). Factual claims are checked against that text. General world knowledge is **not** treated as brochure fact. Invented figures (maintenance amounts, Tower A dates, etc.) are unsupported claims.

## 9. Multilingual testing

User input is **not** translated. Code-mix examples:

```
Kannada alli location heli.
Location yelli ide?
Mujhe project ke baare mein information chahiye.
3 BHK ka price kya hai?
Sir, site visit Saturday morning possible hai?
```

The same string is sent to every model. Language detection is recorded; expected language is scored only if a golden case exists (for example L010).

## 10. Export format

`/save` or `/export` writes:

```
LLM_Evaluation/output/interactive_runs/<run_id>/
  conversation.json
  model_responses.jsonl
  evaluations.jsonl
  session_summary.csv
  session_summary.xlsx
```

Excel sheets:

1. **Conversation** — Run ID, Turn, Timestamp, Customer Message
2. **Responses** — model, provider, answer, intent, action, stage, language, latency, tokens, error
3. **Evaluation** — per-turn objective metrics
4. **Session Summary** — independent aggregates per model

## 11. Dry-run

```bash
python -m LLM_Evaluation.interactive.cli --dry-run
```

Zero external API calls. Verifies configuration, model loading, prompt construction, session init, command parsing, evaluation, and export.

## 12. Model check

```bash
python -m LLM_Evaluation.interactive.cli --check-models
```

Reports provider, logical name, configured model ID, availability, and error. Does not substitute another model. Combine with `--dry-run` to skip live probes.

## 13. Production safety

- No Twilio client, no outbound calls, no `/voice` changes
- Production calling pipeline and `.env` are untouched
- Batch LLM-EVAL.1 / 1.1 (`python run_evaluation.py`) remains unchanged
- Optional demo: `python -m LLM_Evaluation.interactive.cli --demo` (live APIs unless `--dry-run`)

## Example session

```
Customer > Hi, I am interested in Sobha Townpark.

============================================================
TURN 01
CUSTOMER
============================================================

Hi, I am interested in Sobha Townpark.

------------------------------------------------------------
[1] SARVAM — CONVERSATIONAL
------------------------------------------------------------
...
------------------------------------------------------------
[2] SARVAM — FLAGSHIP
------------------------------------------------------------
...
------------------------------------------------------------
[3] CLAUDE — SONNET
------------------------------------------------------------
...
------------------------------------------------------------
[4] CLAUDE — FLAGSHIP
------------------------------------------------------------
...
```
