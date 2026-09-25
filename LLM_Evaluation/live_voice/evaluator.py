"""Assemble a judge packet from a completed live call. Deterministic evidence is not the score."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap
from .constants import JUDGE_CRITERIA
from .judge import LiveVoiceJudge
from .session_store import load_json, run_dir, write_json

bootstrap()

from interactive.outbound_ground_truth import resolve_outbound_ground_truth


def _turns_from_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def deterministic_evidence(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Helper notes only. Never copied into live scores."""
    history: list[dict[str, str]] = []
    notes = []
    for turn in turns:
        if turn.get("speaker") != "customer":
            continue
        utterance = turn.get("corrected_text") or turn.get("text") or ""
        outbound_index = turn.get("outbound_turn_index") or 1
        gold, applicable = resolve_outbound_ground_truth(utterance, int(outbound_index), history)
        history.append({"role": "user", "content": utterance})
        agent = next(
            (
                item
                for item in turns
                if item.get("turn_index", 0) > turn.get("turn_index", 0) and item.get("speaker") == "agent"
            ),
            None,
        )
        notes.append(
            {
                "turn_index": turn.get("turn_index"),
                "outbound_turn_index": outbound_index,
                "ground_truth_applicable": applicable,
                "expected_intent": (gold or {}).get("expected_intent"),
                "expected_action": (gold or {}).get("expected_action"),
                "test_id": (gold or {}).get("test_id"),
                "candidate_declared_intent": (agent or {}).get("candidate_declared_intent"),
                "candidate_declared_action": (agent or {}).get("candidate_declared_action"),
                "note": "candidate_declared_* is untrusted self-report",
            }
        )
    return notes


def build_packet(
    *,
    manifest: dict[str, Any],
    turns: list[dict[str, Any]],
    recording: dict[str, Any] | None,
) -> dict[str, Any]:
    latencies = [t.get("latency_ms") for t in turns if t.get("latency_ms") is not None]
    return {
        "campaign_id": manifest.get("campaign_id"),
        "evaluation_run_id": manifest.get("evaluation_run_id"),
        "alias": manifest.get("alias"),
        "candidate_model_id": manifest.get("model_id"),
        "provider": manifest.get("provider"),
        "call_sid": manifest.get("call_sid"),
        "call_status": manifest.get("call_status"),
        "duration_seconds": manifest.get("duration_seconds"),
        "recording_status": (recording or {}).get("recording_status") or manifest.get("recording_status"),
        "recording_sid": (recording or {}).get("recording_sid") or manifest.get("recording_sid"),
        "recording_available": bool((recording or {}).get("recording_sid") or manifest.get("recording_sid")),
        "reference_scenario": "outbound_baseline_live_voice",
        "transcript": turns,
        "candidate_self_report": [
            {
                "turn_index": t.get("turn_index"),
                "intent": t.get("candidate_declared_intent"),
                "action": t.get("candidate_declared_action"),
                "untrusted": True,
            }
            for t in turns
            if t.get("speaker") == "agent" and (t.get("candidate_declared_intent") or t.get("candidate_declared_action"))
        ],
        "provider_latencies_ms": latencies,
        "deterministic_evidence": deterministic_evidence(turns),
        "instruction": (
            "Score from transcript + ground truth + metadata only. "
            "Do not use candidate_self_report as the score. "
            "Mark unobservable criteria N/A, not zero."
        ),
    }


def evaluate_run_dir(path: Path, judge: LiveVoiceJudge | None = None) -> dict[str, Any]:
    manifest = load_json(path / "manifest.json")
    turns = _turns_from_jsonl(path / "transcript.jsonl")
    recording = None
    rec_path = path / "recording.ref.json"
    if rec_path.exists():
        recording = load_json(rec_path)
    packet = build_packet(manifest=manifest, turns=turns, recording=recording)
    judge = judge or LiveVoiceJudge()
    judged = judge.evaluate_packet(packet)
    output = {
        "campaign_id": manifest.get("campaign_id"),
        "evaluation_run_id": manifest.get("evaluation_run_id"),
        "alias": manifest.get("alias"),
        "model_id": manifest.get("model_id"),
        "call_sid": manifest.get("call_sid"),
        "deterministic_evidence": packet["deterministic_evidence"],
        "candidate_self_report": packet["candidate_self_report"],
        "judge": judged,
        "criteria_names": list(JUDGE_CRITERIA),
        "offline_benchmark_merged": False,
    }
    write_json(path / "evaluator.json", output)
    return output


def evaluate_campaign(campaign_id: str, judge: LiveVoiceJudge | None = None) -> list[dict[str, Any]]:
    from .session_store import campaign_dir

    root = campaign_dir(campaign_id)
    results = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (child / "manifest.json").exists():
            continue
        results.append(evaluate_run_dir(child, judge=judge))
    return results
