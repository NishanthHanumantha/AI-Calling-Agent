# AI Calling Agent - Setup Instructions

## Purpose
This document explains how to set up and run the AI Calling Agent project from a new machine.

---

# Prerequisites

Install the following software before running the project:

1. Python 3.11+
2. Visual Studio Code
3. Git
4. Ngrok
5. Twilio Account
6. SARVAM API Key

---

# Project Files

Important files in this repository:

| File | Purpose |
|--------|--------|
| app_V0.py | Initial prototype with Project Knowledge Base |
| app_v1.py | Introduced RAG from Project Brochure |
| app_v2.py | Improved version of RAG without hallucination |
| app_v3.py | Fixed full cycle - Latest AI Calling Agent version |
| make_call.py | Triggers outbound call |
| leads.txt | Lead data |
| sobha-townpark-brochure.pdf | Project brochure used for FAQ retrieval |
| README.md | Project overview |
| Setup_Instructions.md | Detailed setup guide |
| AICallingAgent_VersionTracker.xlsx | Version tracking log |

---

# Step 1 – Clone Repository

Open Command Prompt:

```bash
git clone <repository-url>
cd ai-calling-agent
```

---

# Step 2 – Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

---

# Step 3 – Install Dependencies

Install required packages:

```bash
pip install fastapi
pip install uvicorn
pip install twilio
pip install openai
pip install python-dotenv
pip install pypdf
pip install faiss-cpu
pip install numpy
pip install pandas
```

Or install from requirements file if available:

```bash
pip install -r requirements.txt
```

---

# Step 4 – Configure Environment Variables

Create a file named:

```text
.env
```

Example:

```env
SARVAM_API_KEY=your_api_key

TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number

PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

Replace all values with actual credentials.

---

# Step 5 – Start FastAPI Server

Open Command Prompt:

```bash
cd ai-calling-agent
```

Run:

```bash
uvicorn app_v3:app --host 0.0.0.0 --port 8000
```

Expected output:

```text
Uvicorn running on http://0.0.0.0:8000
```

Keep this window open.

---

# Step 6 – Start Ngrok

Open a second Command Prompt.

Run:

```bash
ngrok http 8000
```

Example output:

```text
https://abcd1234.ngrok-free.app
```

Copy this URL.

---

# Step 7 – Update Twilio Webhook

Login to Twilio Console.

Navigate to:

Phone Numbers
→ Active Numbers
→ Select Number

Voice Configuration:

```text
Webhook URL:
https://abcd1234.ngrok-free.app/incoming-call
```

Save changes.

---

# Step 8 – Update Application URL

If PUBLIC_URL is used inside code:

Update:

```env
PUBLIC_URL=https://abcd1234.ngrok-free.app
```

Restart FastAPI after changes.

---

# Step 9 – Run Outbound Call

Open third Command Prompt.

Navigate to project:

```bash
cd ai-calling-agent
```

Run:

```bash
python make_call.py
```

This initiates a call through Twilio.

---

# Call Flow

Current conversation flow:

1. Greeting
2. Qualification
3. FAQ Response using brochure
4. Slot Proposal
5. Slot Confirmation
6. Closing

---

# Troubleshooting

## FastAPI not starting

Check:

```bash
pip install fastapi uvicorn
```

---

## Ngrok URL expired

Restart ngrok:

```bash
ngrok http 8000
```

Update Twilio webhook with new URL.

---

## Twilio call not connecting

Verify:

- Twilio balance available
- Correct Account SID
- Correct Auth Token
- Correct Twilio Number
- Valid webhook URL

---

## OpenAI errors

Verify:

```env
OPENAI_API_KEY
```

is present and valid.

---

## Port already in use

Change port:

```bash
uvicorn app_v3:app --host 0.0.0.0 --port 8001
```

Then:

```bash
ngrok http 8001
```

---

# Typical Execution Sequence

Terminal 1:

```bash
venv\Scripts\activate
uvicorn app_v3:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
ngrok http 8000
```

Terminal 3:

```bash
python make_call.py
```

---

# Project Owner

Nishanth Bilimagga Hanumantha

AI Calling Agent Prototype
Sobha Limited