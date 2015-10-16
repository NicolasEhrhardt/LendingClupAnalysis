import pandas as pd
import configparser
from lendingclubapi import LendingClubAPI
from analysis import pre_processing

# Login credentials.
config = configparser.ConfigParser()
config.read("creds_nico.cfg")
investor_id = config.getint("Credentials", "investor_id")
API_key = config.get("Credentials", "API_key")

lc = LendingClubAPI(investor_id, API_key)

notes = lc.get_detailed_notes_owned()
df = pd.DataFrame.from_dict(notes)

pre_processing(df, period=60)

gdf = df.groupby(["periodsSinceIssued", "gradeLetter"]).agg({
    'gainReceivedPerPeriod': ["sum", "mean"],
    'principalReceivedPerPeriod': ["sum", "mean"],
})

imgdf = gdf["gainReceivedPerPeriod"]["mean"]
ndf = imgdf.unstack("gradeLetter")
ndf.plot()

