# AI Calling Agent - Setup Instructions

## Purpose
This document explains how to set up and run the AI Calling Agent, and how the current Lightsail deployment differs from the earlier local development setup.

The application code and Git history are the source of truth. Details that are not in this repository are listed under "Requires confirmation" instead of being assumed.

---

# Current checkpoint

Commit `0aa297a85397a3feb9d4ace483ef0747effe964d` (`0aa297a`), message "Update Twilio voice and STT configuration", is the verified voice and speech-recognition change. It changes only `gather_block()` in `app_v3.py`.

The Gather element emitted by that function is:

| Attribute | Value |
|-----------|--------|
| `action` | `https://s-ai-calling-agent.duckdns.org/handle-speech` |
| `method` | `POST` |
| `speechTimeout` | `5` |
| `speechModel` | `experimental_utterances` |
| `language` | `en-IN` |
| `input` | `speech` |
| `actionOnEmptyResult` | `true` |
| `timeout` | `30` (`GATHER_TIMEOUT`) |

`speechTimeout` is written directly as `"5"` in the TwiML. `app_v3.py` also defines `GATHER_SPEECH_TIMEOUT = "5"`, and that constant is not referenced by `gather_block()`. Changing the constant alone does not change the Gather element.

Spoken prompts use `<Say voice="...">`. The voice id comes from the environment variable `TWILIO_SAY_VOICE`. If it is unset, the code default is `Polly.Aditi`. The `<Say>` element does not set a language attribute.

`app_v3.py` loads configuration with `load_dotenv()` and reads `SARVAM_API_KEY` and `TWILIO_SAY_VOICE` through `os.getenv`. Do not put secret values in this document or in Git.

---

# Two environments

## Current Lightsail deployment

What this repository verifies:

- The speech callback is an absolute HTTPS URL on `https://s-ai-calling-agent.duckdns.org/handle-speech`.
- `app_v3.py` exposes that path as `POST /handle-speech`.
- The call starts at `GET` or `POST /voice` (`app_v3.py`).
- `make_call.py` places the outbound call at `{PUBLIC_URL}/voice`. `PUBLIC_URL` is read from the environment. The Gather `action` is not built from `PUBLIC_URL`; it is the absolute URL above.
- Dependencies declared for this application are in `requirements.txt`: `fastapi`, `uvicorn`, `python-multipart`, `pymupdf`, `requests`, `python-dotenv`, and `twilio`.
- `.env` and `venv/` are listed in `.gitignore`.

The Lightsail instance login method, project directory, virtual environment path, process supervisor, and reverse-proxy configuration are not defined in this repository. See "Requires confirmation" before restarting or changing the live service.

## Previous local development setup

The steps later in this document clone the repository on a workstation, create a `venv`, and expose port `8000` with ngrok. That was the local development path. It is not the current Lightsail deployment.

On the local path, `PUBLIC_URL` must be the ngrok HTTPS origin, and the Twilio voice webhook used by `make_call.py` is `{PUBLIC_URL}/voice`. The Gather `action` in `app_v3.py` still points at the Lightsail HTTPS URL above, so a purely local ngrok session does not receive speech callbacks unless that action is changed. This document does not change application code.

---

# Prerequisites

Install the following before a new local setup:

1. Python 3.11+
2. Visual Studio Code or Cursor
3. Git
4. Ngrok, for the local development path only
5. Twilio account
6. Sarvam API key, stored only in `.env`

---

# Project Files

Important files in this repository:

| File | Purpose |
|--------|--------|
| app_V0.py | Initial prototype with Project Knowledge Base |
| app_v1.py | Introduced RAG from Project Brochure |
| app_v2.py | Improved version of RAG without hallucination |
| app_v3.py | Current calling application, including Gather voice/STT settings |
| make_call.py | Triggers an outbound call to `{PUBLIC_URL}/voice` |
| requirements.txt | Declared application dependencies |
| leads.txt | Lead data |
| sobha-townpark-brochure.pdf | Project brochure used for FAQ retrieval |
| README.md | Project overview |
| Setup_Instructions.md | This setup guide |
| AICallingAgent_VersionTracker.xlsx | Version tracking log |

---

# Git: Lightsail, GitHub, and Cursor

There are three separate copies of this repository:

1. The Lightsail working tree.
2. GitHub remote `origin`: `https://github.com/NishanthHanumantha/AI-Calling-Agent.git`.
3. The Cursor clone on a workstation.

A commit made on Lightsail is not in Cursor until it has been pushed to GitHub and pulled. `main` is the branch used for this project.

Check the copy you are in before fetching, pulling, or pushing:

```bash
git status -sb
git rev-parse --short HEAD
git remote -v
```

Bring a clone up to date with GitHub. `--ff-only` refuses the pull when it would create a merge commit:

```bash
git fetch origin
git pull --ff-only origin main
```

After a commit is already on that machine, publish it:

```bash
git status -sb
git push origin main
```

GitHub HTTPS authentication does not accept the GitHub account password. When Git asks for a password, use a Personal Access Token with access to this repository. Create and store that token in GitHub and in the machine's credential helper. Do not paste the token into this document, into chat, into a commit, or into a command that would be saved in shell history.

Do not force-push `main`. Do not commit `.env`, credential files, recordings, or call transcripts.

---

# Step 1 – Clone Repository

Local workstation:

```bash
git clone https://github.com/NishanthHanumantha/AI-Calling-Agent.git
cd AI-Calling-Agent
```

The folder name after clone is `AI-Calling-Agent`.

---

# Step 2 – Create Virtual Environment

Local workstation:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux

```bash
source venv/bin/activate
```

Whether Lightsail already has a virtual environment, and where it is, requires confirmation. Do not install these packages into the system Python on the server until that environment is identified.

---

# Step 3 – Install Dependencies

From the activated environment:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is the dependency list for `app_v3.py` and `make_call.py`. The separate `LLM_Evaluation/requirements.txt` file is for the evaluation tools and is not required to run the calling application.

---

# Step 4 – Configure Environment Variables

Create a local `.env` file. It is gitignored. Set values only on the machine that runs the app. This document lists names, not values.

Names read by `app_v3.py`:

```text
SARVAM_API_KEY
TWILIO_SAY_VOICE
```

`TWILIO_SAY_VOICE` is optional. The code default is `Polly.Aditi`.

Names required by `make_call.py` before it will place a call:

```text
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_PHONE_NUMBER
TWILIO_TO_NUMBER
PUBLIC_URL
```

`PUBLIC_URL` is the HTTPS origin with no path, for example the ngrok origin on a local machine. `make_call.py` appends `/voice`.

Confirm that required names are set without printing their values:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); names=('SARVAM_API_KEY','TWILIO_ACCOUNT_SID','TWILIO_AUTH_TOKEN','TWILIO_PHONE_NUMBER','TWILIO_TO_NUMBER','PUBLIC_URL'); print('\n'.join(n + '=' + ('set' if os.getenv(n) else 'missing') for n in names))"
```

Expected result: each required name prints as `set`. `TWILIO_SAY_VOICE` may print as `missing` and the application will use `Polly.Aditi`.

---

# Step 5 – Start FastAPI Server (local)

Open a terminal in the repository with the virtual environment active:

```bash
uvicorn app_v3:app --host 0.0.0.0 --port 8000
```

Expected output:

```text
Uvicorn running on http://0.0.0.0:8000
```

Keep this window open.

How the process is started on Lightsail is not in this repository. Do not assume this `uvicorn` command is the live service command.

---

# Step 6 – Start Ngrok (local only)

Open a second terminal.

```bash
ngrok http 8000
```

Copy the HTTPS forwarding URL. That origin is the local `PUBLIC_URL`. Ngrok is not part of the verified Lightsail speech URL.

---

# Step 7 – Twilio voice webhook

`make_call.py` does not read a webhook from the Twilio console. It passes this URL on the outbound call:

```text
{PUBLIC_URL}/voice
```

For the current Lightsail deployment, `PUBLIC_URL` on that machine must be the public HTTPS origin that reaches the running `app_v3.py`. The Gather speech callback is fixed in code:

```text
POST https://s-ai-calling-agent.duckdns.org/handle-speech
```

`app_v3.py` has no `/incoming-call` route. An older local note that used `/incoming-call` does not match this application.

The phone number's webhook inside the Twilio console is not stored in this repository and requires confirmation.

---

# Step 8 – Run Outbound Call

`make_call.py` creates a real Twilio call. Run it only when a test call is intended.

From the repository, with the virtual environment active and `.env` configured:

```bash
python make_call.py
```

Expected output:

```text
Call SID: <Twilio call SID>
```

The script prints the call SID only.

---

# Call Flow

Current conversation flow in `app_v3.py`:

1. Greeting on `/voice`
2. Qualification
3. FAQ response using the brochure
4. Visit-day and slot handling
5. Closing

---

# Troubleshooting

## FastAPI not starting

Install the declared dependencies into the active virtual environment:

```bash
python -m pip install -r requirements.txt
```

## Ngrok URL expired (local only)

Restart ngrok and set `PUBLIC_URL` in `.env` to the new HTTPS origin. Restart the local server after changing `.env`. Speech results still post to the absolute Gather URL in `app_v3.py`, not to the new ngrok origin.

## Twilio call not connecting

Check, without printing secret values:

- The environment-variable check in Step 4 shows the Twilio names as `set`
- Twilio account balance is available
- `PUBLIC_URL` is an HTTPS origin that reaches this application
- The call URL is `{PUBLIC_URL}/voice`
- The running `app_v3.py` is the revision that contains commit `0aa297a`

## Sarvam errors

`app_v3.py` reads `SARVAM_API_KEY` from the environment. The model id in code is `sarvam-105b-conversations`. This application does not read `OPENAI_API_KEY`.

## Port already in use (local)

Use another local port and point ngrok at that port:

```bash
uvicorn app_v3:app --host 0.0.0.0 --port 8001
ngrok http 8001
```

Do not change the Lightsail listen port from this document. That port is not recorded here.

---

# Typical local execution sequence

Terminal 1, application:

```bash
venv\Scripts\activate
uvicorn app_v3:app --host 0.0.0.0 --port 8000
```

On Linux, activate with `source venv/bin/activate` instead of `venv\Scripts\activate`.

Terminal 2, local tunnel:

```bash
ngrok http 8000
```

Terminal 3, outbound call, only when a live test is intended:

```bash
python make_call.py
```

A second terminal for live Lightsail logs requires the confirmed log command for that host. This repository does not contain that command.

---

# Requires confirmation

These items are not defined by the files in this repository:

- Lightsail browser SSH steps and the project directory on the instance
- The virtual environment path on Lightsail
- The process supervisor, service name, restart command, and log command
- The reverse proxy in front of the application, and which port it forwards to
- The Twilio console voice webhook on the phone number
- The value of `PUBLIC_URL` on Lightsail (the name is required; the value must not be copied into Git)

Confirm those on the server before restarting the live application or placing a test call.

---

# Project Owner

Nishanth Bilimagga Hanumantha

AI Calling Agent Prototype
Sobha Limited
