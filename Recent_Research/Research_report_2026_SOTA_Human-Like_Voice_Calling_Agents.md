# Researcher: 2026 SOTA Human-Like Voice Calling Agents & Real-Time Dialogue

We found **4 research papers** on state-of-the-art human-like voice calling agents, streaming speech-to-speech architectures, and expressive conversational synthesis from 2026, organized by type below.

---

## 📄 Research (4)

| # | Title | Source | Date |
|---|-------|--------|------|
| 1 | [DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization](https://arxiv.org/abs/2603.09180) | arXiv (cs.CL) | Mar 2026 |
| 2 | [Interactive TTS: Dynamic Speaking Style Adaptation for Expressive Speech Synthesis](https://arxiv.org/abs/2609.25707) | arXiv (eess.AS) | Sep 2026 |
| 3 | [BayLing-Duplex: Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM](https://arxiv.org/abs/2606.14528) | arXiv (cs.CL) | Jun 2026 |
| 4 | [AdaptDuplex: From Static to Adaptive Full-Duplex Spoken Dialogue](https://arxiv.org/html/2609.29217v1) | arXiv (cs.CL) | Sep 2026 |

**(1)** Re-engineers the traditional cascaded ASR-LLM-TTS pipeline (the exact architecture common in Python/Cursor backends) into a VAD-free streaming flow with chunk-wise micro-turns and special control tokens, removing turn-taking latency while preserving full LLM reasoning.

**(2)** Addresses unnatural, monotonic reading tones in multi-turn interactive calls by explicitly steering dynamic speaking styles (cadence, urgency, warmth) and using Context-Aware Direct Preference Optimization (CADPO) to prevent voice drift.

**(3)** Proposes an end-to-end full-duplex Speech-Language Model that eliminates separate turn-taking detectors entirely, enabling a single autoregressive LLM to listen, speak, hesitate, and handle barge-in interruptions simultaneously.

**(4)** Introduces a token-level adaptive dual-stream protocol for foundation omni models with non-blocking cognitive consolidation, achieving top scores on realistic human-recorded dialogue benchmarks (HumDial-FDBench).

---

## Key Themes Across Sources

- **Micro-Turn & VAD-Free Streaming for Cascaded Pipelines**: Demonstrates that modular architectures (ASR + LLM + TTS) can overcome high turn latencies by breaking conversations into continuous streaming micro-chunks rather than waiting for silence-based VAD boundaries.
- **Dynamic Contextual Prosody & Expressiveness**: Human-like calling requires emotional and contextual adaptation across multi-turn dialogues, preventing repetitive cadence or robotic delivery through preference-aligned TTS conditioning.
- **Native Full-Duplex Turn Coordination**: Shifting turn-taking and interruption handling directly into autoregressive tokens enables models to react smoothly to caller interruptions and overlapping speech without awkward pauses.

---

*Research conducted on October 13, 2026.*
