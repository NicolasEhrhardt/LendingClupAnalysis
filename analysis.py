import numpy as np
import pandas as pd


def is_inactive(row):
    return row.isin(("Charged Off", "Default"))


def pre_processing(df, period=1):
    for column in df.columns:
        if "date" in column.lower():
            df[column] = pd.to_datetime(df[column])

    # Adding convenience columns.
    df["gradeLetter"] = df["grade"].str[0]
    df["daysSinceIssued"] = (pd.datetime.now() - df["issueDate"]).dt.days
    df["periodsSinceIssued"] = np.floor(df["daysSinceIssued"] / period)

    df["gainReceived"] = (
        (1 - is_inactive(df["loanStatus"])) * df["interestReceived"] -
        is_inactive(df["loanStatus"]) * df["principalPending"]
    )

    df["gainReceivedPerPeriod"] = df["gainReceived"] / df["periodsSinceIssued"]
    df["interestReceivedPerPeriod"] = df["interestReceived"] / df["periodsSinceIssued"]
    df["principalReceivedPerPeriod"] = df["principalReceived"] / df["periodsSinceIssued"]
