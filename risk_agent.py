"""
risk_agent.py
--------------
This is the "cook #2" step -- the actual AI agent.

The scoring step (score_transactions.py) just says "this looks weird"
using math. It can't explain *why* in plain language, and it can't
weigh things the way a human analyst would ("this is a huge chunk of
the account's balance AND it's a transfer out -- that combination is
what real fraud looks like").

This agent takes ONE flagged transaction at a time, sends its details
to an AI model (Gemini 2.5 Flash, via Google's free-tier API), and
asks for:
  1. A verdict: BLOCK / REVIEW / ALLOW
  2. A short plain-English explanation of why

This is "agentic" in the sense that it's not just a single fixed
question-answer -- it's given a role, a specific case to reason about,
and it has to produce a structured decision, similar to how a real
risk analyst would triage a queue of flagged transactions.

Requires a free Gemini API key set as an environment variable:
    Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
Get a free key at: aistudio.google.com/apikey (no card required)

Run this after score_transactions.py: python risk_agent.py
"""

import os
import time
import pandas as pd
from google import genai

INPUT_FILE = "flagged_transactions.csv"
OUTPUT_FILE = "risk_report.csv"
MODEL = "gemini-2.5-flash"  # fast, capable, and covered by the free tier

client = genai.Client()  # reads GEMINI_API_KEY from the environment automatically

SYSTEM_PROMPT = """You are a payments risk analyst. You review flagged transactions \
one at a time and decide: BLOCK, REVIEW, or ALLOW.

BLOCK: clear signs of fraud (near-total balance drain, unusual pattern for the type).
REVIEW: suspicious but not conclusive -- needs a human to look closer.
ALLOW: flagged by the automated scorer, but on inspection looks like normal behavior.

Always reply in this exact format, nothing else:
VERDICT: <BLOCK/REVIEW/ALLOW>
REASON: <one or two plain-English sentences a non-technical person could understand>
"""


def build_case_description(row: pd.Series) -> str:
    balance_drained_pct = 0
    if row["oldbalanceOrg"] > 0:
        balance_drained_pct = round(
            (row["oldbalanceOrg"] - row["newbalanceOrig"]) / row["oldbalanceOrg"] * 100, 1
        )

    return f"""Transaction type: {row['type']}
Amount: {row['amount']}
Sender balance before: {row['oldbalanceOrg']}
Sender balance after: {row['newbalanceOrig']}
Percent of sender's balance moved: {balance_drained_pct}%
Receiver balance before: {row['oldbalanceDest']}
Receiver balance after: {row['newbalanceDest']}
Anomaly score from the statistical model: {row['anomaly_score']:.4f} (lower = more unusual)
"""


def investigate(row: pd.Series) -> dict:
    case = build_case_description(row)

    response = client.models.generate_content(
        model=MODEL,
        contents=f"{SYSTEM_PROMPT}\n\nReview this transaction:\n\n{case}",
    )

    reply = response.text.strip()

    verdict, reason = "REVIEW", reply  # safe fallback if parsing fails
    for line in reply.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {"verdict": verdict, "reason": reason}


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "Set your API key first: $env:GEMINI_API_KEY='your-key-here'"
        )

    df = pd.read_csv(INPUT_FILE)
    results = []

    for idx, row in df.iterrows():
        print(f"Investigating transaction step {row['step']}...")
        outcome = investigate(row)
        results.append({**row.to_dict(), **outcome})
        time.sleep(4)  # stay comfortably under the free tier's per-minute limit

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Wrote {len(out_df)} investigated cases to {OUTPUT_FILE}")
    print(out_df[["step", "type", "amount", "verdict", "reason"]].to_string(index=False))
