"""
score_transactions.py
----------------------
This is the "cook #1" step: it looks at every transaction and scores
how unusual it is, using an Isolation Forest.

Why Isolation Forest specifically?
Most transactions are normal. Fraud is rare and *looks different* from
the crowd. Isolation Forest works by trying to "isolate" each data
point with random splits -- unusual points get isolated in very few
splits (they stand out), normal points take many splits (they're
surrounded by similar points). It doesn't need labeled fraud examples
to work, which matches real life: you rarely have a clean "this was
fraud" label for every past transaction.

Run this after generate_data.py: python score_transactions.py
It writes flagged_transactions.csv -- the transactions the model
thinks are suspicious, ranked by how suspicious.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT_FILE = "transactions.csv"
OUTPUT_FILE = "flagged_transactions.csv"
TOP_N_TO_FLAG = 15  # how many of the most suspicious transactions to send to the AI agent


def score(df: pd.DataFrame) -> pd.DataFrame:
    # Turn the transaction "type" (text) into numbers the model can use
    df = df.copy()
    df["type_encoded"] = df["type"].astype("category").cat.codes

    features = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type_encoded",
    ]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # our rough guess at what % of data is anomalous
        random_state=42,
    )
    model.fit(df[features])

    # decision_function: higher = more normal, lower = more unusual
    df["anomaly_score"] = model.decision_function(df[features])
    # predict: -1 means "flagged as anomaly", 1 means "normal"
    df["flagged"] = model.predict(df[features]) == -1

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    scored = score(df)

    flagged = scored[scored["flagged"]].sort_values("anomaly_score").head(TOP_N_TO_FLAG)

    print(f"Scored {len(scored)} transactions.")
    print(f"Flagged {scored['flagged'].sum()} as anomalies.")
    print(f"Sending the top {len(flagged)} most suspicious to the AI risk agent.")

    flagged.to_csv(OUTPUT_FILE, index=False)
