from twilio.rest import Client

account_sid = "SID"

auth_token = "TOKEN"

client = Client(account_sid, auth_token)

call = client.calls.create(
    to="CUSTOMER NUMBER",
    from_="DIALER NUMBER",
    url="https://candied-attest-varying.ngrok-free.dev/voice"
)

print("Call SID:", call.sid)