# ElevenLabs and Cartesia

Checked 12 October 2026. This note is research only. The localhost page in this folder still speaks with Sarvam Bulbul, speaker Ishita. No ElevenLabs key and no Cartesia key are configured, and neither voice is called.

These two are a later trial. Bulbul stays the first speech trial because the caller and the transcription already use Sarvam, on the key this project already has. ElevenLabs and Cartesia are different companies. Each needs its own key, its own voice id, and a separate account.

## What the public listening test measures

The figures below are from the Artificial Analysis Provider Voice Arena, read on 12 October 2026:

https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice

Listeners hear two clips of the same text, each in that provider's own voice, and pick the one that sounds more natural. The score is an Elo rating. A higher score means listeners preferred that model more often. The page says the accent filters are US and UK. It is not an Indian English test, and it is not a test of SOBHA Townpark lines.

Sarvam Bulbul and Microsoft Neerja do not appear on that board. These scores cannot be read as a result against Ishita or Neerja.

| Model | Elo | Samples | Released | Listed price |
| --- | --- | --- | --- | --- |
| Eleven v4 Turbo | 1327 ± 18 | 1,768 | Sept 2026 | $40 / 1M characters |
| Eleven v4 | 1323 ± 17 | 2,057 | Sept 2026 | $80 / 1M characters |
| Sonic 3.6 (Cartesia) | 1284 ± 15 | 2,435 | Aug 2026 | $49 / 1M characters |
| Eleven Flash v2.5 | 1077 ± 11 | 5,700 | Dec 2024 | $50 / 1M characters |
| Polly Neural | 896 ± 14 | 2,873 | Jul 2019 | $16 / 1M characters |

The price column is the leaderboard's own "API pricing" note: cost to generate 1 million characters on the creator's API at that model's default settings. It is not a quote for this project.

Eleven v4 Turbo leads the board. Sonic 3.6 is fourth. Both sit well above ElevenLabs Flash v2.5. Flash is the model ElevenLabs documents for the lowest latency, and it ranks much lower on this listening test. A later trial that picks Flash for speed would not be trying the model listeners preferred.

The intervals overlap between Eleven v4 Turbo and Eleven v4, so the board does not separate those two. Sonic 3.6's interval does not overlap Eleven v4 Turbo's.

## Speed

The published speed numbers are not one measurement, and they are not measured from this computer.

ElevenLabs says Flash models take about 75 milliseconds of model inference, and that this figure excludes the network. For Flash over a websocket, their latency guide gives South Asia a time-to-first-byte of 150–200 milliseconds. Regions listed there are the USA, the Netherlands, and Singapore. India data residency is described as an enterprise option, not the default.

https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization

Cartesia says Sonic produces the first audio in under 90 milliseconds. That is a model-side claim. An independent Speko run dated 28 August 2026 measured Sonic 3.6 at a 120 millisecond median from the US East, on a warm connection. A different index, Coval, over the 30 days ending 1 October 2026, measured a 347.9 millisecond median time-to-first-audio for Sonic 3.6. Those two measurements are not the same path.

https://www.cartesia.ai/sonic

https://benchmarks.speko.ai/tts/cartesia-sonic-3-6/

https://ttsforvoiceagents.com/tts-api/cartesia-sonic-3-6/

This page does not start playback until the whole clip has arrived. A faster model shortens the wait only after the speak path plays the first audio chunk while the rest is still arriving. Until then, the scorecard's voice time is the full clip, for Bulbul and for any later vendor.

## Fit for this call

Cartesia documents Sonic 3.6 (`sonic-3.6`) with an Indian English locale, `en-IN`, and with Hindi, Tamil, Kannada, and the other Indian languages used in later versions of this project. It also documents Hinglish, including Hindi written in Latin script. The request takes a voice id, not a speaker name. Server calls use an `X-API-Key` or `Authorization: Bearer` header. Streaming is a websocket at `wss://api.cartesia.ai/tts/websocket`, or HTTP at `/tts/bytes`.

https://docs.cartesia.ai/build-with-cartesia/tts-models/latest

ElevenLabs documents Indian-accent voices in its library, and Hindi and Tamil among the languages of the multilingual and Flash models. The listening-test leaders are Eleven v4 and Eleven v4 Turbo. The low-latency model id in their own guide is `eleven_flash_v2_5`. A voice is selected by voice id. Server calls use an `xi-api-key` header on `https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`. Their India page offers Indian accents, Hinglish, and India data residency for enterprise accounts.

https://elevenlabs.io/india

Neither vendor is wired, because the key is new and the voice for an Indian English sales call has not been heard on a Townpark sentence. The public board cannot choose that voice: its accent filter is US and UK.

## What a later trial would change

Only the spoken reply on this localhost page. The caller stays `sarvam-105b-conversations`. Transcription stays Sarvam `saaras:v4`. The phone voice stays Polly Kajal-Neural, and that path is not in this folder.

One vendor at a time, after its key is present. The first clip should be a Townpark sentence that contains SOBHA, a BHK, and a square-foot range, heard against the same sentence from Ishita on the Version 5 page. The scorecard already records voice time, so the same sentence can be compared in seconds. If the new call fails, the page should still speak with the voice this copy already has.
