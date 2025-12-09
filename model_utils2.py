import pandas as pd

def compute_rule_columns(df):
    """
    Compute rule-based dropout risk score (0 to 1).
    Updated for CGPA-based data (0–10 scale).

    Uses:
      - attendance (%)
      - avg_cgpa (0–10)
      - fee_weight (0–1)
      - optional: previous_cgpa, backlogs/attempts
    """


    def attendance_weight(att):
        if att >= 90:
            return 0.0
        elif att <= 40:
            return 1.0
        return (90 - att) / 50.0  


    def cgpa_weight(cgpa):
        if cgpa >= 8.0:
            return 0.0
        elif cgpa <= 4.0:
            return 1.0
        return (8.0 - cgpa) / 4.0 


    if "previous_cgpa" in df.columns:
        df["cgpa_change"] = df["avg_cgpa"] - df["previous_cgpa"]

        def cgpa_change_weight(change):
            if change >= 0:
                return 0.0
            elif change <= -2.0:
                return 1.0
            return abs(change) / 2.0

        df["cgpa_change_risk"] = df["cgpa_change"].apply(cgpa_change_weight)
    else:
        df["cgpa_change_risk"] = 0.0


    if "backlogs" in df.columns:
        df["backlog_risk"] = df["backlogs"].apply(lambda x: min(x / 4.0, 1.0))
    elif "attempts" in df.columns:
        df["backlog_risk"] = df["attempts"].apply(lambda x: min(x / 4.0, 1.0))
    else:
        df["backlog_risk"] = 0.0


    df["attendance_risk"] = df["attendance"].apply(attendance_weight)
    df["cgpa_risk"]       = df["avg_cgpa"].apply(cgpa_weight)
    df["fee_risk"]        = df["fee_weight"]


    df["rule_score"] = (
        0.40 * df["attendance_risk"] +
        0.35 * df["cgpa_risk"] +
        0.15 * df["fee_risk"] +
        0.05 * df["cgpa_change_risk"] +
        0.05 * df["backlog_risk"]
    ).clip(0, 1)

    df["rule_flag"] = (df["rule_score"] >= 0.6).astype(int)


    for col in ["attendance_risk", "cgpa_risk", "fee_risk",
                "cgpa_change_risk", "cgpa_change", "backlog_risk"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df
