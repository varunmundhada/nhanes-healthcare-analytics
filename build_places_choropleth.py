"""
Task 3: CDC PLACES state-level obesity/diabetes, aggregated from real county-level
PLACES data (population-weighted mean) since PLACES publishes no state-level file.
"""
import json
import pandas as pd

with open("data/places_county_raw.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
print(f"Raw county rows: {len(df)}")
df["totalpopulation"] = pd.to_numeric(df["totalpopulation"], errors="coerce")
df["obesity_crudeprev"] = pd.to_numeric(df.get("obesity_crudeprev"), errors="coerce")
df["diabetes_crudeprev"] = pd.to_numeric(df.get("diabetes_crudeprev"), errors="coerce")

before = len(df)
df = df.dropna(subset=["totalpopulation"])
print(f"Dropped {before - len(df)} rows with missing population")

def wavg(g, col):
    s = g.dropna(subset=[col])
    if s.empty or s["totalpopulation"].sum() == 0:
        return None
    return (s[col] * s["totalpopulation"]).sum() / s["totalpopulation"].sum()

rows = []
for (abbr, name), g in df.groupby(["stateabbr", "statedesc"]):
    rows.append({
        "state_abbr": abbr, "state_name": name,
        "n_counties": len(g),
        "n_counties_with_obesity": g["obesity_crudeprev"].notna().sum(),
        "n_counties_with_diabetes": g["diabetes_crudeprev"].notna().sum(),
        "obesity_crudeprev_popwt": wavg(g, "obesity_crudeprev"),
        "diabetes_crudeprev_popwt": wavg(g, "diabetes_crudeprev"),
        "total_population": g["totalpopulation"].sum(),
    })

state_df = pd.DataFrame(rows).sort_values("state_name")
state_df.to_csv("out/places_state_aggregated.csv", index=False)
print(f"\nWrote out/places_state_aggregated.csv ({len(state_df)} states/territories)")
print(state_df[["state_abbr", "obesity_crudeprev_popwt", "diabetes_crudeprev_popwt", "n_counties"]].to_string(index=False))
