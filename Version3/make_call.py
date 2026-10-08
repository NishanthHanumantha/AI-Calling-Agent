import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

REQUIRED_VARS = (
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
    "TWILIO_TO_NUMBER",
    "PUBLIC_URL",
)

missing = [name for name in REQUIRED_VARS if not os.getenv(name)]
if missing:
    raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL")

voice_url = PUBLIC_URL.rstrip("/") + "/voice"

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

call = client.calls.create(
    to=TWILIO_TO_NUMBER,
    from_=TWILIO_PHONE_NUMBER,
    url=voice_url,
)

print("Call SID:", call.sid)
