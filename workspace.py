import configparser

import pandas as pd

from lendingclubapi import LendingClubStub
from preprocessing_apidata import pre_processing
from preprocessing_offlinedata import get_data

# Login credentials.
config = configparser.ConfigParser()
config.read("creds_nico.cfg")
investor_id = config.getint("Credentials", "investor_id")
API_key = config.get("Credentials", "API_key")

lc = LendingClubStub(investor_id, API_key)

notes = lc.get_detailed_notes_owned()
api_df = pd.DataFrame.from_dict(notes)

pre_processing(api_df, period=30)

# api_df.boxplot(by="periodsSinceIssued", column="gainPerPeriod")

offline_df = get_data()
loan_df = offline_df[offline_df["id"].isin(api_df["loanId"])]

