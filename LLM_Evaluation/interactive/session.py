from __future__ import annotations

import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluator.dataset import load_dataset
from evaluator.runner import (
    check_models as probe_models,
    load_env,
    load_text,
    load_yaml_config,
    new_run_id,
    root_dir,
    select_models,
)
from models import build_provider
from models.prompting import build_user_payload
from openpyxl import Workbook
from openpyxl.styles import Font

from .conversation import SHARED_PROJECT_CONTEXT, infer_stage
from .evaluator import evaluate_turn, session_summary
from .renderer import redact
from .schemas import DISPLAY_ORDER, DISPLAY_NAMES, ModelSlot, TurnResponse

LOGGER = logging.getLogger("llm_eval.interactive")


def bootstrap_path() -> Path:
    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    return eval_root


def _safe_error(message: str | None) -> str | None:
    if not message:
        return None
    return redact(message)


class InteractiveSession:
    def __init__(
        self,
        dry_run: bool = False,
        max_display_chars: int = 0,
        config_path: Path | None = None,
        providers: dict[str, Any] | None = None,
    ):
        bootstrap_path()
        load_env()
        self.dry_run = dry_run
        self.max_display_chars = max_display_chars
        self.eval_root = root_dir()
        self.config = load_yaml_config(config_path)
        self.system_prompt = load_text(self.eval_root / "prompts" / "calling_agent_system_prompt.txt")
        self.retrieved_context = SHARED_PROJECT_CONTEXT
        dataset_path = self.eval_root / "dataset" / "golden_dataset.csv"
        self.dataset = load_dataset(str(dataset_path))
        selected, skipped = select_models(self.config, ["all"])
        self.skip_notes = skipped
        by_name = {item["name"]: item for item in selected}
        self.slots: list[ModelSlot] = []
        self._runtimes: dict[str, dict[str, Any]] = {}
        self._providers: dict[str, Any] = providers or {}
        for alias in DISPLAY_ORDER:
            runtime = by_name.get(alias)
            if runtime is None:
                self.slots.append(
                    ModelSlot(
                        alias=alias,
                        provider=alias.split("_")[0],
                        role="",
                        model_id="",
                        available=False,
                        status="UNAVAILABLE",
                        error=f"not in active configuration ({'; '.join(skipped) or 'unknown'})",
                    )
                )
                continue
            available = bool(runtime.get("available"))
            error = None
            if not runtime.get("api_key"):
                available = False
                error = f"missing {runtime.get('api_key_env')}"
            elif not runtime.get("model"):
                available = False
                error = f"missing model id ({runtime.get('model_env')})"
            status = "AVAILABLE" if available else "UNAVAILABLE"
            slot = ModelSlot(
                alias=alias,
                provider=runtime["provider"],
                role=runtime.get("role") or "",
                model_id=runtime.get("model") or "",
                available=available,
                status=status,
                error=error,
                base_url=runtime.get("base_url") or "",
                runtime=runtime,
            )
            self.slots.append(slot)
            self._runtimes[alias] = runtime
            if providers is None and available and not dry_run:
                self._providers[alias] = build_provider(
                    runtime["provider"],
                    api_key=runtime["api_key"],
                    model=runtime["model"],
                    base_url=runtime["base_url"],
                    timeout=runtime["timeout"],
                    temperature=runtime["temperature"],
                    max_tokens=runtime["max_tokens"],
                    max_retries=runtime["max_retries"],
                    model_alias=alias,
                    model_role=runtime.get("role"),
                )
        self.run_id = new_run_id()
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.histories: dict[str, list[dict[str, str]]] = {alias: [] for alias in DISPLAY_ORDER}
        self.turns: list[dict[str, Any]] = []
        self.evaluations: list[dict[str, Any]] = []
        self.last_debug: dict[str, Any] = {}
        self.exported_dir: Path | None = None

    def reset(self) -> None:
        self.histories = {alias: [] for alias in DISPLAY_ORDER}
        self.turns = []
        self.evaluations = []
        self.last_debug = {}

    def next_turn_index(self) -> int:
        return len(self.turns) + 1

    def generate_turn(self, utterance: str) -> list[TurnResponse]:
        turn_index = self.next_turn_index()
        extra = {"conversation_stage": infer_stage(utterance, turn_index), "stage": infer_stage(utterance, turn_index)}
        histories_snapshot = {alias: list(items) for alias, items in self.histories.items()}
        payloads = {}
        for alias in DISPLAY_ORDER:
            payloads[alias] = {
                "system_prompt": self.system_prompt,
                "conversation_history": histories_snapshot[alias],
                "customer_utterance": utterance,
                "retrieved_context": self.retrieved_context,
                "extra": extra,
            }
        self.last_debug = {
            "user_message": utterance,
            "retrieved_knowledge": self.retrieved_context,
            "system_instructions_summary": self.system_prompt.split("\n", 1)[0][:240],
            "conversation_stage": extra["conversation_stage"],
            "same_user_message": True,
            "same_retrieved_knowledge": True,
            "same_system_prompt": True,
            "per_model_history": {alias: list(histories_snapshot[alias]) for alias in DISPLAY_ORDER},
            "model_request_metadata": {
                alias: {
                    "provider": slot.provider,
                    "model_id": slot.model_id,
                    "status": slot.status,
                    "max_tokens": (slot.runtime or {}).get("max_tokens"),
                    "timeout": (slot.runtime or {}).get("timeout"),
                }
                for slot in self.slots
                for alias in [slot.alias]
            },
        }
        responses: dict[str, TurnResponse] = {}
        if self.dry_run:
            for slot in self.slots:
                responses[slot.alias] = TurnResponse(
                    alias=slot.alias,
                    provider=slot.provider,
                    model_id=slot.model_id,
                    answer=None,
                    intent=None,
                    action=None,
                    language=None,
                    stage=None,
                    raw_response="",
                    latency_ms=None,
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    error_type="DRY_RUN",
                    error_message="Dry-run: no API call made",
                    schema_valid=None,
                )
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(self._call_model, slot, payloads[slot.alias]): slot.alias for slot in self.slots
                }
                for future in as_completed(futures):
                    alias = futures[future]
                    try:
                        responses[alias] = future.result()
                    except Exception as exc:  # isolation: one model must not kill the turn
                        slot = next(s for s in self.slots if s.alias == alias)
                        responses[alias] = TurnResponse(
                            alias=alias,
                            provider=slot.provider,
                            model_id=slot.model_id,
                            answer=None,
                            intent=None,
                            action=None,
                            language=None,
                            stage=None,
                            raw_response="",
                            latency_ms=None,
                            input_tokens=None,
                            output_tokens=None,
                            total_tokens=None,
                            error_type="API_ERROR",
                            error_message=_safe_error(str(exc)),
                            schema_valid=None,
                        )
        ordered = [responses[alias] for alias in DISPLAY_ORDER]
        for resp in ordered:
            self.histories[resp.alias].append({"role": "user", "content": utterance})
            if not resp.error_type:
                self.histories[resp.alias].append(
                    {"role": "assistant", "content": resp.answer or resp.raw_response or ""}
                )
        self.turns.append(
            {
                "index": turn_index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "utterance": utterance,
                "responses": [asdict(item) for item in ordered],
                "histories_before": histories_snapshot,
            }
        )
        return ordered

    def _call_model(self, slot: ModelSlot, payload: dict[str, Any]) -> TurnResponse:
        if slot.alias not in self._providers:
            return TurnResponse(
                alias=slot.alias,
                provider=slot.provider,
                model_id=slot.model_id,
                answer=None,
                intent=None,
                action=None,
                language=None,
                stage=None,
                raw_response="",
                latency_ms=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                error_type="UNAVAILABLE",
                error_message=_safe_error(slot.error or "model unavailable"),
                schema_valid=None,
            )
        result = self._providers[slot.alias].generate(**payload)
        usage = result.get("usage") or {}
        error_type = result.get("error_type")
        return TurnResponse(
            alias=slot.alias,
            provider=slot.provider,
            model_id=result.get("model_id") or slot.model_id,
            answer=result.get("answer"),
            intent=result.get("intent"),
            action=result.get("action"),
            language=result.get("language"),
            stage=None,
            raw_response=result.get("raw_response") or "",
            latency_ms=result.get("latency_ms"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            error_type=error_type,
            error_message=_safe_error(result.get("error_message") or result.get("error")),
            schema_valid=result.get("schema_valid"),
            confidence=result.get("confidence"),
        )

    def evaluate_latest(self) -> dict[str, Any] | None:
        if not self.turns:
            return None
        turn = self.turns[-1]
        responses = [TurnResponse(**item) for item in turn["responses"]]
        prior = turn.get("histories_before") or {alias: [] for alias in DISPLAY_ORDER}
        ev = evaluate_turn(
            turn["utterance"],
            responses,
            self.dataset,
            prior,
            turn["index"],
            self.retrieved_context,
        )
        self.evaluations = [item for item in self.evaluations if item.get("turn_index") != turn["index"]]
        self.evaluations.append(ev)
        return ev

    def preview_payloads(self, utterance: str) -> dict[str, str]:
        """Build semantically identical user payloads without calling APIs."""
        extra = {"conversation_stage": infer_stage(utterance, self.next_turn_index())}
        return {
            alias: build_user_payload(self.histories[alias], utterance, self.retrieved_context, extra)
            for alias in DISPLAY_ORDER
        }

    def export(self, output_root: Path | None = None) -> Path:
        root = output_root or (self.eval_root / "output" / "interactive_runs" / self.run_id)
        root.mkdir(parents=True, exist_ok=True)
        conversation = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "mode": "OFFLINE / INTERACTIVE EVALUATION",
            "twilio": "DISABLED",
            "retrieved_context": self.retrieved_context,
            "models": [
                {
                    "alias": slot.alias,
                    "provider": slot.provider,
                    "model_id": slot.model_id,
                    "status": slot.status,
                }
                for slot in self.slots
            ],
            "turns": [{"run_id": self.run_id, "turn": t["index"], "timestamp": t["timestamp"], "customer_message": t["utterance"]} for t in self.turns],
        }
        (root / "conversation.json").write_text(json.dumps(conversation, indent=2, ensure_ascii=False), encoding="utf-8")
        with (root / "model_responses.jsonl").open("w", encoding="utf-8") as handle:
            for turn in self.turns:
                for resp in turn["responses"]:
                    row = {"run_id": self.run_id, "turn": turn["index"], "timestamp": turn["timestamp"], **resp}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (root / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
            for ev in self.evaluations:
                handle.write(json.dumps({"run_id": self.run_id, **ev}, ensure_ascii=False, default=str) + "\n")
        summary = session_summary(self.evaluations)
        csv_path = root / "session_summary.csv"
        fieldnames = ["metric"] + list(DISPLAY_ORDER)
        metrics = [
            ("intent_accuracy", lambda s: s["intent"]["accuracy"]),
            ("intent_precision", lambda s: s["intent"]["precision"]),
            ("intent_recall", lambda s: s["intent"]["recall"]),
            ("intent_macro_f1", lambda s: s["intent"]["f1"]),
            ("action_accuracy", lambda s: s["action"]["accuracy"]),
            ("action_macro_f1", lambda s: s["action"]["f1"]),
            ("factual_accuracy", lambda s: s["factual_accuracy"]),
            ("grounded_pct", lambda s: s["grounded_pct"]),
            ("unsupported_claim_rate", lambda s: s["unsupported_rate"]),
            ("context_accuracy", lambda s: s["context_accuracy"]),
            ("stage_accuracy", lambda s: s["stage_accuracy"]),
            ("avg_relevance", lambda s: s["avg_relevance"]),
            ("avg_completeness", lambda s: s["avg_completeness"]),
            ("avg_clarity", lambda s: s["avg_clarity"]),
            ("avg_conversational", lambda s: s["avg_conversational"]),
            ("avg_latency", lambda s: s["avg_latency"]),
            ("p50_latency", lambda s: s["p50_latency"]),
            ("p95_latency", lambda s: s["p95_latency"]),
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for name, getter in metrics:
                writer.writerow({"metric": name, **{alias: getter(summary[alias]) for alias in DISPLAY_ORDER}})
        self._write_excel(root / "session_summary.xlsx", summary)
        self.exported_dir = root
        return root

    def _write_excel(self, path: Path, summary: dict[str, dict[str, Any]]) -> None:
        wb = Workbook()
        conv = wb.active
        conv.title = "Conversation"
        conv.append(["Run ID", "Turn", "Timestamp", "Customer Message"])
        for cell in conv[1]:
            cell.font = Font(bold=True)
        for turn in self.turns:
            conv.append([self.run_id, turn["index"], turn["timestamp"], turn["utterance"]])

        resp_ws = wb.create_sheet("Responses")
        resp_ws.append(
            [
                "Run ID",
                "Turn",
                "Model",
                "Provider",
                "Response",
                "Intent",
                "Action",
                "Stage",
                "Language",
                "Latency",
                "Input Tokens",
                "Output Tokens",
                "Error",
            ]
        )
        for cell in resp_ws[1]:
            cell.font = Font(bold=True)
        for turn in self.turns:
            for resp in turn["responses"]:
                resp_ws.append(
                    [
                        self.run_id,
                        turn["index"],
                        DISPLAY_NAMES.get(resp["alias"], resp["alias"]),
                        resp.get("provider"),
                        resp.get("answer") or resp.get("raw_response"),
                        resp.get("intent"),
                        resp.get("action"),
                        resp.get("stage"),
                        resp.get("language"),
                        resp.get("latency_ms"),
                        resp.get("input_tokens"),
                        resp.get("output_tokens"),
                        _safe_error(resp.get("error_message") or resp.get("error_type")),
                    ]
                )

        ev_ws = wb.create_sheet("Evaluation")
        ev_ws.append(
            [
                "Run ID",
                "Turn",
                "Model",
                "Intent Correct",
                "Precision",
                "Recall",
                "F1",
                "Action Correct",
                "Factual Accuracy",
                "Grounded",
                "Unsupported Claims",
                "Hallucination",
                "Context Correct",
                "Stage Correct",
                "Relevance",
                "Completeness",
                "Clarity",
                "Conversational Quality",
            ]
        )
        for cell in ev_ws[1]:
            cell.font = Font(bold=True)
        for ev in self.evaluations:
            for model in ev.get("models") or []:
                ctx = bool(model.get("context_used")) and not model.get("context_error")
                ev_ws.append(
                    [
                        self.run_id,
                        ev.get("turn_index"),
                        DISPLAY_NAMES.get(model.get("alias"), model.get("alias")),
                        model.get("intent_correct"),
                        None,
                        None,
                        None,
                        model.get("action_correct"),
                        model.get("factual_accuracy"),
                        model.get("grounded"),
                        json.dumps(model.get("unsupported_claims") or [], ensure_ascii=False),
                        model.get("hallucination"),
                        ctx if not model.get("api_error") else None,
                        model.get("stage_correct"),
                        model.get("relevance"),
                        model.get("completeness"),
                        model.get("clarity"),
                        model.get("conversational"),
                    ]
                )

        sum_ws = wb.create_sheet("Session Summary")
        sum_ws.append(["Metric"] + [DISPLAY_NAMES[a] for a in DISPLAY_ORDER])
        for cell in sum_ws[1]:
            cell.font = Font(bold=True)
        rows = [
            ("Intent Accuracy", lambda s: s["intent"]["accuracy"]),
            ("Intent Precision", lambda s: s["intent"]["precision"]),
            ("Intent Recall", lambda s: s["intent"]["recall"]),
            ("Intent Macro F1", lambda s: s["intent"]["f1"]),
            ("Action Accuracy", lambda s: s["action"]["accuracy"]),
            ("Action Macro F1", lambda s: s["action"]["f1"]),
            ("Factual Accuracy", lambda s: s["factual_accuracy"]),
            ("Grounded Response %", lambda s: s["grounded_pct"]),
            ("Unsupported Claim Rate", lambda s: s["unsupported_rate"]),
            ("Context Accuracy", lambda s: s["context_accuracy"]),
            ("Stage Accuracy", lambda s: s["stage_accuracy"]),
            ("Avg Relevance", lambda s: s["avg_relevance"]),
            ("Avg Completeness", lambda s: s["avg_completeness"]),
            ("Avg Clarity", lambda s: s["avg_clarity"]),
            ("Avg Conversational Quality", lambda s: s["avg_conversational"]),
            ("Avg Latency", lambda s: s["avg_latency"]),
            ("P50 Latency", lambda s: s["p50_latency"]),
            ("P95 Latency", lambda s: s["p95_latency"]),
        ]
        for label, getter in rows:
            sum_ws.append([label] + [getter(summary[alias]) for alias in DISPLAY_ORDER])
        wb.save(path)

    def check_models(self) -> dict[str, Any]:
        if self.dry_run:
            return {
                "check_models": True,
                "dry_run": True,
                "results": [
                    {
                        "alias": slot.alias,
                        "provider": slot.provider,
                        "model_id": slot.model_id,
                        "status": "DRY_RUN",
                        "error_message": None,
                    }
                    for slot in self.slots
                ],
            }
        return probe_models()


def format_check_report(result: dict[str, Any]) -> str:
    lines = ["MODEL CHECK", "=" * 60]
    for row in result.get("results") or []:
        lines.append(f"Provider: {row.get('provider')}")
        lines.append(f"Logical Model Name: {row.get('alias')}")
        lines.append(f"Configured Model ID: {row.get('model_id')}")
        lines.append(f"API availability: {row.get('status')}")
        if row.get("auth"):
            lines.append(f"Authentication: {row.get('auth')}")
        if row.get("model_check"):
            lines.append(f"Model check: {row.get('model_check')}")
        if row.get("error_type") or row.get("error_message"):
            lines.append(f"Error: {redact(row.get('error_message') or row.get('error_type'))}")
        lines.append("-" * 40)
    if result.get("dry_run"):
        lines.append("Dry-run: no external API calls.")
    return "\n".join(lines)
