from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from models import build_provider
from models.availability import check_claude_model, check_sarvam_model
from models.parsing import ALLOWED_ACTIONS, ALLOWED_INTENTS

from .aggregation import (
    category_intent_accuracy,
    error_analysis,
    macro_f1,
    model_test_matrix,
    provider_comparisons,
    summarize,
    summarize_models,
)
from .dataset import load_dataset
from .deterministic import evaluate_facts, evaluate_intent_action
from .excel import write_workbook
from .grounding import evaluate_context_handling, evaluate_grounding, evaluate_language
from .response_quality import maybe_run_judge, score_response_quality

EVAL_VERSION = "1.1.0"
LOGGER = logging.getLogger("llm_eval")
IN_SCOPE_PROVIDERS = ("sarvam", "claude")
ACTIVE_ALIASES = (
    "sarvam_conversational",
    "sarvam_flagship",
    "claude_flagship",
    "claude_sonnet",
)

DETAIL_COLUMNS = [
    "run_id",
    "test_id",
    "category",
    "language",
    "provider",
    "model_alias",
    "model_id",
    "model_role",
    "conversation_stage",
    "customer_utterance",
    "expected_intent",
    "predicted_intent",
    "intent_pass",
    "expected_action",
    "predicted_action",
    "action_pass",
    "expected_facts",
    "missing_facts",
    "fact_accuracy",
    "grounded",
    "hallucination",
    "unsupported_claims",
    "context_handling",
    "language_pass",
    "schema_valid",
    "relevance_score",
    "clarity_score",
    "completeness_score",
    "conversation_quality_score",
    "stage_appropriateness_score",
    "response_word_count",
    "overall_response_quality",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost",
    "error_type",
    "error_message",
    "acceptable_answer_criteria",
    "answer",
    "raw_response",
]


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def load_env() -> None:
    eval_root = root_dir()
    load_dotenv(eval_root.parent / ".env")
    load_dotenv(eval_root / ".env", override=False)


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or (root_dir() / "config" / "models.yaml")
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def resolve_model_runtime(name: str, spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    api_key = (os.getenv(spec.get("api_key_env", ""), "") or "").strip()
    model = (os.getenv(spec.get("model_env", ""), "") or spec.get("default_model") or "").strip()
    base_url = (os.getenv(spec.get("base_url_env", ""), "") or spec.get("default_base_url") or "").strip()
    provider = spec.get("provider", name)
    enabled = bool(spec.get("enabled", True))
    in_scope = provider in IN_SCOPE_PROVIDERS and enabled
    return {
        "name": name,
        "provider": provider,
        "role": spec.get("role"),
        "enabled": enabled,
        "in_scope": in_scope,
        "api_key": api_key,
        "model": model,
        "model_id": model,
        "model_alias": name,
        "model_role": spec.get("role"),
        "base_url": base_url,
        "models_url": spec.get("models_url") or "https://api.anthropic.com/v1/models",
        "timeout": int(config.get("timeout_seconds", 30)),
        "temperature": config.get("temperature", 0.1),
        "max_tokens": config.get("max_tokens", 400),
        "max_retries": int(config.get("max_retries", 2)),
        "available": bool(in_scope and api_key and model and base_url),
        "api_key_env": spec.get("api_key_env"),
        "model_env": spec.get("model_env"),
        "config_source": spec.get("model_env") or "default_model",
    }


def requested_aliases(config: dict[str, Any], models: list[str] | None, provider: str | None) -> list[str]:
    specs = config.get("models") or {}
    active = list(config.get("active_model_aliases") or ACTIVE_ALIASES)
    if provider:
        return [alias for alias in active if (specs.get(alias) or {}).get("provider") == provider]
    if not models:
        return active
    if models == ["all"]:
        return active
    return models


def select_models(
    config: dict[str, Any],
    requested: list[str] | None,
    provider: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    specs = config.get("models") or {}
    names = requested_aliases(config, requested, provider)
    selected = []
    skipped = []
    for name in names:
        if name not in specs:
            skipped.append(f"{name}: unknown model alias")
            continue
        runtime = resolve_model_runtime(name, specs[name], config)
        if runtime["provider"] not in IN_SCOPE_PROVIDERS:
            skipped.append(f"{name}: OUT OF SCOPE")
            continue
        if not runtime["enabled"]:
            skipped.append(f"{name}: disabled in config")
            continue
        selected.append(runtime)
        if not runtime["available"]:
            skipped.append(f"{name}: missing {runtime['api_key_env']} or model/base URL")
    return selected, skipped


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def evaluate_one(case: dict[str, Any], generation: dict[str, Any], judge_fn=None, judge_prompt: str | None = None) -> dict[str, Any]:
    api_failed = bool(generation.get("error_type") or generation.get("error"))
    if api_failed:
        usage = generation.get("usage") or {}
        return {
            "run_id": generation.get("run_id"),
            "test_id": case["test_id"],
            "category": case["category"],
            "language": case["language"],
            "provider": generation.get("provider"),
            "model_alias": generation.get("model_alias"),
            "model_id": generation.get("model_id") or generation.get("model"),
            "model_role": generation.get("model_role"),
            "conversation_stage": case.get("conversation_stage"),
            "customer_utterance": case.get("customer_utterance"),
            "expected_intent": case.get("expected_intent"),
            "predicted_intent": None,
            "intent_pass": None,
            "expected_action": case.get("expected_action"),
            "predicted_action": None,
            "action_pass": None,
            "expected_facts": case.get("expected_facts") or [],
            "missing_facts": None,
            "fact_accuracy": None,
            "grounded": None,
            "hallucination": None,
            "unsupported_claims": None,
            "context_handling": None,
            "language_pass": None,
            "schema_valid": None,
            "relevance_score": None,
            "clarity_score": None,
            "completeness_score": None,
            "conversation_quality_score": None,
            "stage_appropriateness_score": None,
            "response_word_count": None,
            "overall_response_quality": None,
            "latency_ms": generation.get("latency_ms"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_cost": "NOT_CONFIGURED",
            "error": generation.get("error"),
            "error_type": generation.get("error_type") or "API_ERROR",
            "error_message": generation.get("error_message") or generation.get("error"),
            "acceptable_answer_criteria": case.get("acceptable_answer_criteria"),
            "answer": None,
            "raw_response": generation.get("raw_response"),
            "confidence": None,
        }

    intent_action = evaluate_intent_action(case, generation)
    facts = evaluate_facts(case.get("expected_facts") or [], generation.get("answer"))
    grounding = evaluate_grounding(
        case.get("retrieved_context") or "",
        generation.get("answer"),
        case.get("expected_action"),
    )
    language = evaluate_language(case, generation)
    context = evaluate_context_handling(case, generation.get("answer"), generation.get("action"))
    quality = score_response_quality(case, generation, grounding)
    quality.update(maybe_run_judge(case, generation, judge_fn, judge_prompt))
    usage = generation.get("usage") or {}
    return {
        **intent_action,
        **facts,
        **grounding,
        **language,
        **quality,
        "run_id": generation.get("run_id"),
        "test_id": case["test_id"],
        "category": case["category"],
        "language": case["language"],
        "provider": generation.get("provider"),
        "model": generation.get("model"),
        "model_alias": generation.get("model_alias"),
        "model_id": generation.get("model_id") or generation.get("model"),
        "model_role": generation.get("model_role"),
        "conversation_stage": case.get("conversation_stage"),
        "customer_utterance": case.get("customer_utterance"),
        "expected_facts": facts["facts_expected"],
        "schema_valid": bool(generation.get("schema_valid")),
        "context_handling": context,
        "latency_ms": generation.get("latency_ms"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "estimated_cost": "NOT_CONFIGURED",
        "error": None,
        "error_type": None,
        "error_message": None,
        "acceptable_answer_criteria": case.get("acceptable_answer_criteria"),
        "answer": generation.get("answer"),
        "raw_response": generation.get("raw_response"),
        "confidence": generation.get("confidence"),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for key in ("expected_facts", "missing_facts", "facts_found", "unsupported_claims"):
                if key in serialized and not isinstance(serialized[key], str):
                    serialized[key] = json.dumps(serialized[key], ensure_ascii=False)
            writer.writerow({col: serialized.get(col) for col in DETAIL_COLUMNS})


def format_check_models(results: list[dict[str, Any]]) -> str:
    lines = ["MODEL AVAILABILITY", ""]
    lines.append("SARVAM")
    lines.append("-" * 48)
    for row in results:
        if row["provider"] == "sarvam":
            lines.append(f"{row['alias']:<24} {row['status']}")
            if row.get("auth"):
                lines.append(f"  AUTHENTICATION = {row['auth']}")
            if row.get("model_check"):
                lines.append(f"  MODEL = {row['model_check']}")
            if row.get("error_type"):
                lines.append(f"  error_type = {row['error_type']}")
            if row.get("available_model_ids") and row["status"] != "AVAILABLE":
                lines.append("  discovered_ids = " + ", ".join(row["available_model_ids"][:12]))
    lines += ["", "CLAUDE", "-" * 48]
    for row in results:
        if row["provider"] == "claude":
            lines.append(f"{row['alias']:<24} {row['status']}")
            if row.get("auth"):
                lines.append(f"  AUTHENTICATION = {row['auth']}")
            if row.get("model_check"):
                lines.append(f"  MODEL = {row['model_check']}")
            if row.get("error_type"):
                lines.append(f"  error_type = {row['error_type']}")
            if row.get("available_model_ids") and row["status"] != "AVAILABLE":
                lines.append("  discovered_ids = " + ", ".join(row["available_model_ids"][:12]))
    lines += ["", "QWEN", "-" * 48, "OUT OF SCOPE — NOT TESTED", ""]
    return "\n".join(lines)


def check_models(config_path: Path | None = None) -> dict[str, Any]:
    load_env()
    config = load_yaml_config(config_path)
    selected, skipped = select_models(config, ["all"])
    results = []
    for spec in selected:
        if spec["provider"] == "claude":
            probe = check_claude_model(spec["api_key"], spec["model"], spec["models_url"], timeout=spec["timeout"])
        elif spec["provider"] == "sarvam":
            probe = check_sarvam_model(spec["api_key"], spec["model"], spec["base_url"], timeout=spec["timeout"])
        else:
            continue
        results.append(
            {
                "alias": spec["name"],
                "provider": spec["provider"],
                "model_id": spec["model"],
                "role": spec["role"],
                "status": probe["status"],
                "auth": probe.get("auth"),
                "model_check": probe.get("model"),
                "error_type": probe.get("error_type"),
                "error_message": probe.get("error_message"),
                "available_model_ids": probe.get("available_model_ids") or [],
            }
        )
    text = format_check_models(results)
    LOGGER.info("\n%s", text)
    return {
        "check_models": True,
        "eval_version": EVAL_VERSION,
        "results": results,
        "skip_notes": skipped,
        "report": text,
        "qwen": "OUT OF SCOPE — NOT TESTED",
    }


def dry_run_report(
    dataset: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    skipped: list[str],
    system_prompt: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    sarvam = [item for item in selected if item["provider"] == "sarvam"]
    claude = [item for item in selected if item["provider"] == "claude"]
    return {
        "dry_run": True,
        "eval_version": EVAL_VERSION,
        "providers_in_scope": ["Sarvam", "Claude"],
        "qwen": "OUT OF SCOPE",
        "models_configured": 4,
        "sarvam_models": 2,
        "claude_models": 2,
        "dataset_cases": len(dataset),
        "golden_cases": 10,
        "potential_benchmark_executions": 4 * len(dataset),
        "test_ids": [case["test_id"] for case in dataset],
        "allowed_intents": sorted(ALLOWED_INTENTS),
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "prompt_chars": len(system_prompt),
        "models_configured_list": [item["name"] for item in selected],
        "models_that_would_run": [item["name"] for item in selected if item["available"]],
        "models_unavailable": [item["name"] for item in selected if not item["available"]],
        "skip_notes": skipped,
        "api_calls": 0,
        "dataset_version": config.get("dataset_version"),
        "prompt_version": config.get("prompt_version"),
        "sarvam_count": len(sarvam),
        "claude_count": len(claude),
    }


def run_benchmark(
    models: list[str] | None = None,
    dataset_path: Path | None = None,
    dry_run: bool = False,
    config_path: Path | None = None,
    provider: str | None = None,
    check_only: bool = False,
) -> dict[str, Any]:
    load_env()
    eval_root = root_dir()
    config = load_yaml_config(config_path)
    if check_only:
        return check_models(config_path)

    dataset_file = dataset_path or (eval_root / "dataset" / "golden_dataset.csv")
    cases = load_dataset(str(dataset_file))
    system_prompt = load_text(eval_root / "prompts" / "calling_agent_system_prompt.txt")
    judge_prompt = load_text(eval_root / "prompts" / "evaluator_prompt.txt")
    selected, skipped = select_models(config, models, provider=provider)

    if dry_run:
        report = dry_run_report(cases, selected, skipped, system_prompt, config)
        LOGGER.info("DRY-RUN: zero API calls")
        LOGGER.info("Providers in scope: Sarvam, Claude")
        LOGGER.info("Qwen: OUT OF SCOPE")
        LOGGER.info("Models configured: 4 (Sarvam 2, Claude 2)")
        LOGGER.info("Golden cases: %s", len(cases))
        LOGGER.info("Potential benchmark executions: %s", 4 * len(cases))
        for line in skipped:
            LOGGER.info("skip: %s", line)
        return report

    run_id = new_run_id()
    out_dir = eval_root / "output" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    runnable = [spec for spec in selected if spec["available"]]
    providers = []
    for spec in runnable:
        providers.append(
            (
                spec,
                build_provider(
                    spec["provider"],
                    api_key=spec["api_key"],
                    model=spec["model"],
                    base_url=spec["base_url"],
                    timeout=spec["timeout"],
                    temperature=spec["temperature"],
                    max_tokens=spec["max_tokens"],
                    max_retries=spec["max_retries"],
                    model_alias=spec["model_alias"],
                    model_role=spec["model_role"],
                ),
            )
        )

    raw_records: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    successful = 0
    failed = 0

    for case in cases:
        extra = {"conversation_stage": case["conversation_stage"]}
        for spec, provider_obj in providers:
            LOGGER.info("Running %s (%s) on %s", spec["name"], spec["model"], case["test_id"])
            try:
                generation = provider_obj.generate(
                    system_prompt=system_prompt,
                    conversation_history=case["conversation_history"],
                    customer_utterance=case["customer_utterance"],
                    retrieved_context=case["retrieved_context"],
                    extra=extra,
                )
            except Exception as exc:
                generation = {
                    "model": spec["model"],
                    "model_id": spec["model"],
                    "model_alias": spec["model_alias"],
                    "model_role": spec["model_role"],
                    "provider": spec["provider"],
                    "intent": None,
                    "language": None,
                    "answer": None,
                    "action": None,
                    "confidence": None,
                    "raw_response": "",
                    "usage": {},
                    "latency_ms": 0,
                    "error": f"{type(exc).__name__}: provider failed",
                    "error_type": "OTHER_API_ERROR",
                    "error_message": type(exc).__name__,
                    "schema_valid": False,
                }
            generation["run_id"] = run_id
            generation["model_alias"] = generation.get("model_alias") or spec["model_alias"]
            generation["model_role"] = generation.get("model_role") or spec["model_role"]
            generation["model_id"] = generation.get("model_id") or spec["model"]
            if generation.get("error_type") or generation.get("error"):
                failed += 1
            else:
                successful += 1
            raw_records.append(
                {
                    "timestamp": timestamp,
                    "run_id": run_id,
                    "eval_version": EVAL_VERSION,
                    "dataset_version": config.get("dataset_version"),
                    "prompt_version": config.get("prompt_version"),
                    "test_id": case["test_id"],
                    "provider": generation.get("provider"),
                    "model_alias": generation.get("model_alias"),
                    "model_id": generation.get("model_id"),
                    "model_role": generation.get("model_role"),
                    "input": {
                        "conversation_stage": case["conversation_stage"],
                        "conversation_history": case["conversation_history"],
                        "customer_utterance": case["customer_utterance"],
                        "retrieved_context": case["retrieved_context"],
                    },
                    "output": {
                        "intent": generation.get("intent"),
                        "language": generation.get("language"),
                        "answer": generation.get("answer"),
                        "action": generation.get("action"),
                        "confidence": generation.get("confidence"),
                    },
                    "raw_response": generation.get("raw_response"),
                    "latency_ms": generation.get("latency_ms"),
                    "usage": generation.get("usage"),
                    "error": generation.get("error"),
                    "error_type": generation.get("error_type"),
                    "error_message": generation.get("error_message"),
                }
            )
            detail_rows.append(evaluate_one(case, generation, judge_fn=None, judge_prompt=judge_prompt))

    aliases = [spec["name"] for spec in selected]
    summary_rows = summarize_models(detail_rows, aliases)
    error_rows = error_analysis(detail_rows)
    comparison_rows = provider_comparisons(summary_rows)
    matrix_rows = model_test_matrix(detail_rows, aliases, [case["test_id"] for case in cases])
    config_rows = [
        {
            "Provider": spec["provider"],
            "Model Alias": spec["name"],
            "Model ID": spec["model"],
            "Role": spec["role"],
            "Endpoint": spec["base_url"],
            "Generation Parameters": f"max_tokens={spec['max_tokens']}; temperature omitted for Claude",
            "Thinking Configuration": "not set",
            "Structured Output Support": "prompt-enforced JSON",
            "Availability": "configured" if spec["available"] else "missing credentials/model",
            "Availability Error": None,
            "Configuration Source": spec["config_source"],
        }
        for spec in selected
    ]

    scheduled = len(cases) * len(runnable)
    metadata = {
        "run_id": run_id,
        "evaluation_version": EVAL_VERSION,
        "dataset_version": config.get("dataset_version"),
        "prompt_version": config.get("prompt_version"),
        "providers_in_scope": ["sarvam", "claude"],
        "models_configured": 4,
        "models_available": len(runnable),
        "benchmark_cases": len(cases),
        "scheduled_executions": scheduled,
        "successful_executions": successful,
        "failed_executions": failed,
        "timestamp": timestamp,
        "baseline_preserved": "output/runs/20260918_071117",
        "qwen": "OUT OF SCOPE",
    }

    _write_jsonl(out_dir / "raw_outputs.jsonl", raw_records)
    _write_csv(out_dir / "evaluation_results.csv", detail_rows)
    extra_meta = {
        "run_id": run_id,
        "eval_version": EVAL_VERSION,
        "evaluation_version": EVAL_VERSION,
        "dataset_version": config.get("dataset_version"),
        "prompt_version": config.get("prompt_version"),
        "baseline": "LLM-EVAL.1.1 (does not merge LLM-EVAL.1 20260918_071117)",
    }
    write_workbook(
        out_dir / "evaluation_summary.xlsx",
        summary_rows,
        detail_rows,
        error_rows,
        cases,
        extra_meta=extra_meta,
        comparison_rows=comparison_rows,
        matrix_rows=matrix_rows,
        latency_rows=summary_rows,
        config_rows=config_rows,
    )
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latest_dir = eval_root / "output"
    _write_jsonl(latest_dir / "raw_outputs.jsonl", raw_records)
    _write_csv(latest_dir / "evaluation_results.csv", detail_rows)
    try:
        write_workbook(
            latest_dir / "evaluation_summary.xlsx",
            summary_rows,
            detail_rows,
            error_rows,
            cases,
            extra_meta=extra_meta,
            comparison_rows=comparison_rows,
            matrix_rows=matrix_rows,
            latency_rows=summary_rows,
            config_rows=config_rows,
        )
    except OSError as exc:
        LOGGER.warning("Could not update latest Excel copy: %s", type(exc).__name__)

    return {
        "dry_run": False,
        **metadata,
        "skip_notes": skipped,
        "n_rows": len(detail_rows),
        "intent_macro_f1": macro_f1(detail_rows, "expected_intent", "predicted_intent"),
        "action_macro_f1": macro_f1(detail_rows, "expected_action", "predicted_action"),
        "per_category_intent": category_intent_accuracy(detail_rows),
        "output_dir": str(out_dir),
        "summary": summary_rows,
        "legacy_provider_summary": summarize(detail_rows, sorted({r.get("provider") for r in detail_rows if r.get("provider")})),
    }
