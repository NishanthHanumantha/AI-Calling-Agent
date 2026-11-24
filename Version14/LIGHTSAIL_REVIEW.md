# Stakeholder review on a new Lightsail computer

This is a browser demo of Version 14. It does not ring a mobile, and it does not use the telephony company. The existing DuckDNS voice service stays as it is. Do not restart `ai-calling-agent.service`, and do not point this demo at `s-ai-calling-agent.duckdns.org`.

Use the AWS account that already has the remaining credit. Do not open a second AWS account. The smallest Linux plan is $5 a month. Stop the instance when the review is finished so the credit is not used up.

Tell IT before the link is sent. Load the Sarvam balance first. Each stakeholder call spends that balance.

## The computer

1. In the existing account, create a new Lightsail instance in Mumbai. Choose the smallest Linux plan and a static address.
2. Open only ports 80 and 443.
3. Choose a new hostname. A new DuckDNS name is enough. It must not be the hostname of the current voice service.
4. Point that hostname at the new static address.
5. Copy Version 14 onto the instance. Put the Sarvam key and `STAKEHOLDER_PASSPHRASE` in an uncommitted `.env` on that computer. Do not commit the passphrase or the key.
6. Run Version 14 so it listens only on `127.0.0.1`, port 8021, the same way it does on this PC.
7. Put Caddy on the instance. It answers HTTPS for the new hostname and forwards to `127.0.0.1:8021`. Caddy obtains the certificate. Do not use a Lightsail load balancer. That charge is larger than the remaining credit.

A stakeholder's browser will not open the microphone on a plain `http://` address. The link has to be `https://`.

## What to send

Send the `https://` link and the passphrase. The call page and the dashboard ask for it once in that browser. One stakeholder at a time: two people talking together would mix, because the program keeps a single conversation.

They allow the microphone, talk as a customer would, and hang up. The call appears on the dashboard with the price, the project, and the recording of that browser conversation.

The telephony dialer still sends Record as false. This recording is the browser call, not a telephony recording.

After the review, stop the instance. Connecting a telephony company stays a later step.
