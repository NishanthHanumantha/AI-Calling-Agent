from twilio.rest import Client

account_sid = "AC8aa1a9255159e015d82a4b753cd76ad6"

auth_token = "68ed8b421f4a4ce7ca8c436d16a168ba"

client = Client(account_sid, auth_token)

call = client.calls.create(
    to="+918095841985",
    from_="+15673722697",
    url="https://candied-attest-varying.ngrok-free.dev/voice"
)

print("Call SID:", call.sid)