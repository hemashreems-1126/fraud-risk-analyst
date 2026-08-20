# AI Risk Manager

An AI agent that reviews flagged payment transactions and explains, in
plain English, whether they should be blocked, reviewed, or allowed —
the way a human risk analyst would triage a queue of suspicious cases.

## Why this exists

Payment platforms process huge volumes of transactions. Simple
statistical models can flag transactions that *look* unusual, but they
can't explain *why* in a way a human can act on quickly. This project
adds an AI reasoning layer on top of a statistical flagging step, so
every flagged case comes with a clear verdict and a plain-English
reason, not just a number.

## How it works

1. **`generate_data.py`** — creates a set of realistic mobile-payment
   transactions (amount, balances before/after, transaction type),
   with a small percentage seeded as fraud-like.
2. **`score_transactions.py`** — an Isolation Forest model scores every
   transaction for how statistically unusual it is, and flags the
   top suspicious ones.
3. **`risk_agent.py`** — the AI agent. For each flagged transaction, it
   sends the case details to Gemini 2.5 Flash (via Google's free-tier
   API), which reasons about the pattern (e.g. "94% of the sender's
   balance moved out in one transfer") and returns a verdict: `BLOCK`,
   `REVIEW`, or `ALLOW`, with a short explanation.
4. **`main.py`** — runs all three steps in order.

## Running it

```bash
pip install -r requirements.txt
# Windows PowerShell:
$env:GEMINI_API_KEY="your-key-here"
python main.py
```

Get a free API key (no card required) at aistudio.google.com/apikey.

Final results land in `risk_report.csv` — every investigated
transaction with its verdict and reasoning.

## What broke, and how I fixed it

I originally tried building this as a 5-agent orchestrated system,
where separate agents each checked one signal (amount, location,
device, etc.) and a final orchestrator agent combined their outputs.
The orchestration step — getting each agent's output correctly passed
into the next agent's input without losing context — kept breaking,
and I couldn't fully explain why it worked when it did. I scaled the
design down to one well-understood agent doing focused reasoning on
one case at a time. It's smaller, but I can explain every part of it,
which matters more than looking impressive.

## What I'd build next

- Multiple specialist agents (amount pattern, location pattern, device
  pattern) feeding into one orchestrator — the version I originally
  attempted, now that the single-agent version works reliably as a
  foundation.
- A simple dashboard (Streamlit) instead of a CSV output.
- Real transaction data instead of synthetic data.
