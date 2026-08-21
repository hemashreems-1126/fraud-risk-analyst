"""
risk_agent.py — now using Groq instead of Gemini (Gemini free tier caps at
20 requests/day, which is below our 22 flagged transactions and will always fail).
"""

import os
import time
import pandas as pd
from groq import Groq
import time

INPUT_FILE = "flagged_transactions.csv"
OUTPUT_FILE = "risk_report.csv"
MODEL = "openai/gpt-oss-120b"

client = Groq()  # reads GROQ_API_KEY from the environment automatically

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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this transaction:\n\n{case}"},
        ],
    )

    reply = response.choices[0].message.content.strip()

    verdict, reason = "REVIEW", reply
    for line in reply.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {"verdict": verdict, "reason": reason}


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Set your API key first: $env:GROQ_API_KEY='your-key-here'")

    df = pd.read_csv(INPUT_FILE)
    results = []

    for idx, row in df.iterrows():
        print(f"Investigating transaction step {row['step']}...")
        outcome = investigate(row)
        results.append({**row.to_dict(), **outcome})
        time.sleep(2)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Wrote {len(out_df)} investigated cases to {OUTPUT_FILE}")
    print(out_df[["step", "type", "amount", "verdict", "reason"]].to_string(index=False))
