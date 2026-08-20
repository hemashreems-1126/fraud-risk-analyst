import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def score_transactions():
    df = pd.read_csv("transactions.csv")

    features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    X = df[features]
    y = df["isFraud"]

    # Split — model never sees test labels during training
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.3, random_state=42, stratify=y
    )

    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(X_train)

    # -1 = anomaly, 1 = normal → convert to 1 = flagged fraud, 0 = not
    raw_preds = model.predict(X_test)
    df_test = df_test.copy()
    df_test["flagged"] = (raw_preds == -1).astype(int)
    df_test["anomaly_score"] = model.decision_function(X_test)

    precision = precision_score(y_test, df_test["flagged"])
    recall = recall_score(y_test, df_test["flagged"])
    tn, fp, fn, tp = confusion_matrix(y_test, df_test["flagged"]).ravel()

    print(f"Test set: {len(df_test)} transactions ({y_test.sum()} actual fraud)")
    print(f"Flagged: {df_test['flagged'].sum()} transactions")
    print(f"Precision: {precision:.0%} ({tp} of {df_test['flagged'].sum()} flagged were real fraud)")
    print(f"Recall: {recall:.0%} (caught {tp} of {y_test.sum()} actual fraud cases)")
    print(f"False positives: {fp} — each means a legitimate customer's transaction gets held for review")

    # Save the metrics summary for your README/video
    with open("metrics_summary.txt", "w") as f:
        f.write(f"Test set: {len(df_test)} transactions ({y_test.sum()} actual fraud)\n")
        f.write(f"Flagged: {df_test['flagged'].sum()} transactions\n")
        f.write(f"Precision: {precision:.0%}\n")
        f.write(f"Recall: {recall:.0%}\n")
        f.write(f"False positives: {fp}\n")

    # Save flagged transactions (the ones your agent will explain) as before
    flagged_df = df_test[df_test["flagged"] == 1].sort_values("anomaly_score")
    flagged_df.to_csv("flagged_transactions.csv", index=False)
    return flagged_df

if __name__ == "__main__":
    score_transactions()