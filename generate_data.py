"""
generate_data.py
-----------------
Creates a synthetic set of payment transactions to work with.

Why synthetic data instead of downloading a dataset?
Because you need something you can explain end-to-end, including how
the data itself is shaped. This mimics the structure of the well-known
PaySim mobile-money dataset (type, amount, balances before/after,
a small % of fraud), but we generate it ourselves with numpy so the
whole pipeline has no hidden external dependency.

Run this first: python generate_data.py
It writes transactions.csv into this folder.
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # same "random" data every time we run this -> reproducible

N_TRANSACTIONS = 2000
FRAUD_RATE = 0.03  # ~3% of transactions will be fraudulent

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


def generate_transactions(n=N_TRANSACTIONS):
    rows = []
    for i in range(n):
        is_fraud = np.random.rand() < FRAUD_RATE

        txn_type = np.random.choice(TRANSACTION_TYPES)

        # Normal transactions: small-to-medium amounts, balances make sense
        # Fraudulent transactions: tend to drain an account close to fully,
        # often TRANSFER or CASH_OUT, larger relative to balance.
        old_balance_org = round(np.random.uniform(1000, 100000), 2)

        if is_fraud:
            txn_type = np.random.choice(["TRANSFER", "CASH_OUT"])
            amount = round(old_balance_org * np.random.uniform(0.85, 1.0), 2)
        else:
            amount = round(np.random.uniform(10, old_balance_org * 0.5), 2)

        new_balance_org = max(0, round(old_balance_org - amount, 2))

        old_balance_dest = round(np.random.uniform(0, 50000), 2)
        new_balance_dest = round(old_balance_dest + amount, 2)

        rows.append({
            "step": i,
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": old_balance_org,
            "newbalanceOrig": new_balance_org,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
            "isFraud": int(is_fraud),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("transactions.csv", index=False)
    print(f"Generated {len(df)} transactions, {df['isFraud'].sum()} marked fraud.")
    print("Saved to transactions.csv")
