#!/usr/bin/env python3
"""Local interactive multi-model conversation lab. Makes no Twilio calls."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent
for path in (str(EVAL_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from .commands import parse_command
from .renderer import (
    banner,
    help_text,
    render_debug,
    render_evaluation,
    render_history,
    render_models,
    render_session_summary,
    render_status,
    render_turn,
)
from .schemas import DEMO_TURNS, TurnResponse
from .session import InteractiveSession, format_check_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive four-model LLM comparison lab (local, no Twilio).")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline; zero API calls")
    parser.add_argument("--check-models", action="store_true", help="Probe configured Sarvam and Claude models")
    parser.add_argument("--demo", action="store_true", help="Run a scripted 5-turn demonstration")
    parser.add_argument("--max-display-chars", type=int, default=0, help="Truncate displayed answers (0 = no limit)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def dry_run_report(session: InteractiveSession) -> str:
    lines = [
        "DRY-RUN VERIFICATION",
        "=" * 60,
        "MODE: OFFLINE / INTERACTIVE EVALUATION",
        "TWILIO: DISABLED",
        f"Configuration loaded: {bool(session.config)}",
        f"System prompt loaded: {bool(session.system_prompt)}",
        f"Golden dataset cases: {len(session.dataset)}",
        f"Retrieved knowledge chars: {len(session.retrieved_context)}",
        f"Session run_id: {session.run_id}",
        "",
        "Models:",
    ]
    for slot in session.slots:
        lines.append(f"  {slot.alias}: {slot.provider} / {slot.model_id or '(missing id)'} / {slot.status}")
    sample = "Where is it located?"
    payloads = session.preview_payloads(sample)
    first = next(iter(payloads.values()))
    same_kb = all(session.retrieved_context in text for text in payloads.values())
    same_user = all(sample in text for text in payloads.values())
    lines += [
        "",
        f"Prompt construction OK: {bool(first)}",
        f"Same user message in all payloads: {same_user}",
        f"Same knowledge context in all payloads: {same_kb}",
        f"Command parse /help: {parse_command('/help')}",
        f"Command parse customer text: {parse_command('Hello')}",
    ]
    session.generate_turn(sample)
    ev = session.evaluate_latest()
    export_dir = session.export()
    lines += [
        f"Evaluation pipeline OK: {ev is not None}",
        f"Export pipeline OK: {export_dir.exists()}",
        f"Export dir: {export_dir}",
        "API calls made: 0",
        "=" * 60,
    ]
    return "\n".join(lines)


def handle_command(session: InteractiveSession, command: str, arg: str, out) -> bool:
    if command in {"/help"}:
        out(help_text())
    elif command == "/reset":
        session.reset()
        out("Conversation reset. Next message is TURN 01.")
    elif command == "/history":
        out(render_history(session.turns))
    elif command == "/evaluate":
        if arg.lower() == "session":
            if not session.evaluations and session.turns:
                session.evaluate_latest()
            out(render_session_summary(session.evaluations))
        else:
            ev = session.evaluate_latest()
            if ev is None:
                out("No turns to evaluate.")
            else:
                out(render_evaluation(ev))
    elif command in {"/save", "/export"}:
        path = session.export()
        out(f"Saved session to {path}")
    elif command == "/models":
        out(render_models(session.slots))
    elif command == "/status":
        last = session.turns[-1]["responses"] if session.turns else None
        responses = [TurnResponse(**item) for item in last] if last else None
        out(render_status(session.slots, responses))
    elif command == "/debug":
        out(render_debug(session.last_debug or {"note": "No turn yet."}))
    elif command == "/quit":
        return False
    else:
        out(f"Unknown command: {command}. Type /help")
    return True


def run_demo(session: InteractiveSession, out) -> None:
    out("=" * 60)
    out("DEMO MODE — scripted 5-turn comparison. Not a phone call.")
    out("TWILIO: DISABLED")
    out("=" * 60)
    out(banner())
    for utterance in DEMO_TURNS:
        out(f"Customer > {utterance}")
        responses = session.generate_turn(utterance)
        out(render_turn(session.turns[-1]["index"], utterance, responses, session.max_display_chars))
    ev = session.evaluate_latest()
    if ev:
        out(render_evaluation(ev))
    path = session.export()
    out(f"Demo export: {path}")


def interactive_loop(session: InteractiveSession, out, stdin=None) -> int:
    out(banner())
    stream = stdin or sys.stdin
    while True:
        try:
            out("Customer > ", end="")
            line = stream.readline()
            if line == "":
                break
            raw = line.strip()
        except (EOFError, KeyboardInterrupt):
            out("\nExiting.")
            break
        if not raw:
            continue
        command, arg = parse_command(raw)
        if command:
            if not handle_command(session, command, arg, out):
                out("Goodbye.")
                break
            continue
        responses = session.generate_turn(raw)
        out(render_turn(session.turns[-1]["index"], raw, responses, session.max_display_chars))
    return 0


class Printer:
    def __init__(self, sink=None):
        self.sink = sink or sys.stdout

    def __call__(self, text: str = "", end: str = "\n") -> None:
        self.sink.write(text + end)
        self.sink.flush()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    session = InteractiveSession(dry_run=args.dry_run, max_display_chars=args.max_display_chars)
    out = Printer()
    if args.check_models:
        result = session.check_models()
        out(format_check_report(result))
        return 0
    if args.dry_run and not args.demo:
        out(dry_run_report(session))
        return 0
    if args.demo:
        run_demo(session, out)
        return 0
    return interactive_loop(session, out)


if __name__ == "__main__":
    raise SystemExit(main())
