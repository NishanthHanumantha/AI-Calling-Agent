# AI-Calling-Agent



\## Overview

AI-powered outbound calling agent for real estate lead qualification and appointment scheduling.



\## Features

\- Automated outbound calling using Twilio

\- Speech-to-Text processing

\- AI-powered conversation handling

\- Lead qualification

\- FAQ answering using brochure knowledge base

\- Appointment scheduling

\- Call summary generation



\## Tech Stack

\- Python

\- FastAPI

\- Sarvam AI

\- ngrok



\## Project Files



| File | Description |

|--------|------------|

| app\_V0.py | Initial calling agent prototype |

| app\_v1.py | Brochure FAQ integration |

| app\_v2.py | Brochure FAQ integration with improvement |

| app\_v3.py | Visit Scheduling and full call flow improvement |

| make\_call.py | Outbound call trigger script |

| sobha-townpark-brochure.pdf | Knowledge base document |

| leads.txt | Sample lead data |



\## Call Flow



Greeting

→ Lead Qualification

→ FAQ Handling

→ Visit Proposal

→ Confirmation

→ Call Completion



\## Version History



\### V0

\- Initial FastAPI + Twilio integration

\- Basic AI conversation



\### V1

\- Added brochure-based FAQ responses. But few answers were hallucinated by AI. 



\### V2

\- Added brochure-based FAQ responses. Hallucination was fixed but visit scheduling was breaking. 



\### V3

\- Visit scheduling and entire call flow was completed. STT was also improved.



\## Author

Nishanth Bilimagga Hanumantha

