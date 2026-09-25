"""Live-voice evaluation constants. Candidate aliases match the interactive lab."""

from __future__ import annotations

CANDIDATE_ORDER = (
    "sarvam_conversational",
    "deepseek_flagship",
    "claude_sonnet",
    "claude_flagship",
)

DISPLAY_NAMES = {
    "sarvam_conversational": "Sarvam Conversational",
    "deepseek_flagship": "DeepSeek V4.1 Flash",
    "claude_sonnet": "Claude Sonnet",
    "claude_flagship": "Claude Opus",
}

CONFIRMATION_PHRASE = "LIVE-EVAL YES"
EVAL_BIND = "127.0.0.1"
EVAL_PORT = 8001

TERMINAL_CALL_STATUSES = frozenset(
    {"completed", "failed", "busy", "no-answer", "canceled", "cancelled"}
)

LIVE_SPOKEN_APPENDIX = (
    "LIVE VOICE: The answer field is spoken aloud on a phone call. "
    "Use one or two short sentences. No lists, no markdown, no JSON in the spoken answer string."
)

FIXED_OPENING = (
    "Hi, this is Sobha Limited calling. I'm getting in touch about SOBHA Townpark - "
    "a New York-inspired luxury community near Electronic City on Hosur Road in Bengaluru. "
    "Is now a good time for a quick call?"
)

SAY_VOICE = "Polly.Aditi"
GATHER_TIMEOUT = "30"
SPEECH_HINTS = (
    "amenities, pricing, location, floor plan, configuration, "
    "site visit, yes, no, ten am, eleven thirty am, four pm, six pm"
)

MAX_EMPTY_RETRIES = 3
MAX_TURNS = 24
DEFAULT_POLL_SECONDS = 2.0
DEFAULT_CALL_TIMEOUT_SECONDS = 600.0

# Intended budget for one live-voice LLM section. Not a guaranteed HTTP response time.
# create_subprocess_exec is not cancelled and can overrun this value.
LLM_SECTION_BUDGET_SECONDS = 8.0
FALLBACK_SPOKEN = "Sorry, I missed that for a moment. Could you say that again?"

ERROR_CLASSES = (
    "LLM",
    "STT",
    "TTS",
    "TELEPHONY",
    "NETWORK",
    "OPERATOR_SCRIPT",
    "UNOBSERVABLE",
)

JUDGE_CRITERIA = (
    "intent_accuracy",
    "action_accuracy",
    "factual_accuracy",
    "grounding",
    "unsupported_claims",
    "context_retention",
    "conversation_stage_accuracy",
    "site_visit_completion",
    "response_latency",
    "stt_issues",
    "conversation_quality",
)

DONE_PHRASES = (
    "goodbye",
    "good bye",
    "that's all",
    "thats all",
    "not interested",
    "discuss it with",
    "get back to you",
    "have a great day",
    "no thanks",
)
