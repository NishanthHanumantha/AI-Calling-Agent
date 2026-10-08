**AI Calling Agent — Live Call Setup**

**Current working configuration**



AWS Lightsail · Twilio · Flask · Sarvam AI



1\. Start the Lightsail instance



Confirm the instance is running and connect using the AWS Lightsail browser-based SSH terminal.



2\. Navigate to the project



cd \~/AI-Calling-Agent



3\. Verify the environment



Confirm .env contains the required API credentials and Twilio configuration. Never commit or display the secret values.



4\. Restart the calling service



sudo systemctl restart ai-calling-agent.service

sudo systemctl is-active ai-calling-agent.service



Expected service status: active.



5\. Make an outbound call



cd \~/AI-Calling-Agent

./venv/bin/python make\_call.py



Record the returned Call SID for tracking the test.



6\. Monitor live transcripts



Open a second Lightsail terminal and run:



sudo journalctl -u ai-calling-agent.service -f -o cat



Watch for TWILIO FORM, CUSTOMER SAID (raw), and CUSTOMER SAID (corrected) entries.



7\. Validate the call



Check speech recognition, voice response, conversation flow, and the final call outcome. Confirm that the call completes as expected.





**Deepseek-Flash**

Terminal 1 — DeepSeek server



Keep this terminal running throughout the call:



cd \~/AI-Calling-Agent-live-eval \&\& \\

LLM\_PROVIDER=deepseek \\

LLM\_MODEL=deepseek-flash \\

\~/AI-Calling-Agent/venv/bin/python -m uvicorn app\_v3:app --host 127.0.0.1 --port 8001



You should see:



BROCHURE LOADED

TOTAL CHUNKS: 34

Uvicorn running on http://127.0.0.1:8001

Terminal 2 — Live transcript



Use this terminal to display the customer's speech live:



sudo journalctl -f -o cat | grep -E 'TWILIO FORM|CUSTOMER SAID|SpeechResult|UnstableSpeechResult'



Leave it running while you screen-record.



You'll see:



TWILIO FORM: {... 'SpeechResult': '...'}

CUSTOMER SAID (raw): ...

CUSTOMER SAID (corrected): ...

Terminal 3 — Initiate the DeepSeek call



Run this only when you're ready to start the recording:



cd \~/AI-Calling-Agent-live-eval \&\& \\

LLM\_PROVIDER=deepseek \\

LLM\_MODEL=deepseek-flash \\

\~/AI-Calling-Agent/venv/bin/python make\_call.py











**Claude-Sonnet-4.6**

2\. Switch .env to Claude Sonnet



Run this in any terminal:



cd \~/AI-Calling-Agent-live-eval \&\& \\

cp .env .env.after\_deepseek \&\& \\

sed -i 's/^LLM\_PROVIDER=.\*/LLM\_PROVIDER=claude\_sonnet/' .env \&\& \\

sed -i 's/^LLM\_MODEL=.\*/LLM\_MODEL=claude-sonnet-4-6/' .env \&\& \\

echo "=== ACTIVE MODEL ===" \&\& \\

grep -E '^(LLM\_PROVIDER|LLM\_MODEL)=' .env



Expected:



LLM\_PROVIDER=claude\_sonnet

LLM\_MODEL=claude-sonnet-4-6

3\. Terminal 1 — Start Claude server

cd \~/AI-Calling-Agent-live-eval \&\& \\

LLM\_PROVIDER=claude\_sonnet \\

LLM\_MODEL=claude-sonnet-4-6 \\

\~/AI-Calling-Agent/venv/bin/python -m uvicorn app\_v3:app --host 127.0.0.1 --port 8001



Wait for:



Uvicorn running on http://127.0.0.1:8001

4\. Terminal 2 — Live transcript



Same command as before:



sudo journalctl -f -o cat | grep -E 'TWILIO FORM|CUSTOMER SAID|SpeechResult|UnstableSpeechResult'

5\. Terminal 3 — Start Claude call



When you're ready to record:



cd \~/AI-Calling-Agent-live-eval \&\& \\

LLM\_PROVIDER=claude\_sonnet \\

LLM\_MODEL=claude-sonnet-4-6 \\

\~/AI-Calling-Agent/venv/bin/python make\_call.py

