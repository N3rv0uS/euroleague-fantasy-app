# app_streamlit.py
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ==================== CONFIG ====================
st.set_page_config(page_title="EuroLeague Fantasy – Player Game Logs", layout="wide")
OUT_DIR = Path("out")
DEFAULT_SEASON = "2025"
DEFAULT_COMPETITION = "E"  # E = EuroLeague, U = EuroCup
DEFAULT_MODE = "perGame"

# ---------- (optional) simple password gate ----------
PW_OK = True
if "general" in st.secrets and "password" in st.secrets["general"]:
    st.sidebar.text_input("Password", type="password", key="pw")
    PW_OK = st.session_state.get("pw", "") == st.secrets["general"]["password"]
    if not PW_OK:
        st.stop()

st.title("EuroLeague Fantasy – Player Game Logs")

# ==================== UI CONTROLS ====================
colA, colB, colC, colD = st.columns([1,1,1,2])

with colA:
    season = st.text_input("Season", value=DEFAULT_SEASON)
with colB:
    competition = st.selectbox("Competition", ["E", "U"], index=0)
with colC:
    stat_mode = st.selectbox("Mode", ["perGame"], index=0)
with colD:
    st.caption("Live στο cloud, με απλό password & on-demand ανανέωση δεδομένων")

fpath = OUT_DIR / f"player_gamelogs_{season}_{stat_mode}.csv"

# ---------- Update button (on-demand fetch) ----------
with st.container():
    colU1, colU2 = st.columns([1, 5])
    with colU1:
        do_update = st.button("🔄 Update (fetch gamelogs for ALL players)")
    with colU2:
        st.caption("Τρέχει το fetch script μόνο όταν το ζητήσεις. Αν παίρνει ώρα είναι φυσιολογικό — κατεβαίνουν gamelogs για όλους.")

    if do_update:
        cmd = [
            sys.executable, "fetch_euroleague_stats.py",
            "--kind", "gamelogs",
            "--seasons", str(season),
            "--competition", str(competition),
            "--mode", str(stat_mode),
            "--out", str(OUT_DIR),
        ]
        with st.spinner("Updating data… (fetching gamelogs from Incrowd feeds)"):
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
                st.success("✅ Update completed.")
                st.code(out)
            except subprocess.CalledProcessError as e:
                st.error("❌ Update failed. Δες το log παρακάτω.")
                st.code(e.output or str(e))

st.write("---")

# ==================== LOAD DATA ====================
if not fpath.exists():
    st.info(f"Δεν βρέθηκε αρχείο: **{fpath}**. Πάτησε **Update** για να το δημιουργήσεις/ανανεώσεις.")
    st.stop()

df = pd.read_csv(fpath)

# ==================== COLUMN MAPPING HELPERS ====================
def pick_col(_df: pd.DataFrame, options: List[str]) -> Optional[str]:
    """Return the first existing column from options (case-insensitive)."""
    if not len(_df.columns):
        return None
    colmap = {c.lower(): c for c in _df.columns}
    for opt in options:
        if opt in _df.columns:
            return opt
        lo = opt.lower()
        if lo in colmap:
            return colmap[lo]
    return None

# candidate maps for common stats/fields
CAND = {
    "player_name": ["player_displayName","player_shortName","player_fullName","player_name","player.displayName","player.shortName"],
    "player_code": ["player_code","player.code","code"],
    "game_date":   ["game_gameDate","game_date","gameDate","date"],
    "opponent":    ["opponent_shortName","opposition_shortName","opponent","opposition","game_opponent_shortName"],
    "home_away":   ["game_homeAway","homeAway","isHome","homeaway"],
    "team_name":   ["team_shortName","team_name","team.shortName","team"],

    "MIN": ["MIN","minutes","statistics_minutes"],
    "PTS": ["PTS", "points", "statistics_points"],
    "REB": ["REB","rebounds","statistics_reboundsTotal","statistics_rebounds"],
    "AST": ["AST","assists","statistics_assists"],
    "PIR": ["PIR","indexRating","efficiency","statistics_indexRating"],
    "TOV": ["TOV","turnovers","statistics_turnovers"],
    "FDR": ["FDR","foulsDrawn","statistics_foulsDrawn"],
    "FGM": ["FGM","fieldGoalsMade","statistics_fieldGoalsMade"],
    "FGA": ["FGA","fieldGoalsAttempted","statistics_fieldGoalsAttempted"],
    "TPM": ["3PM","threePointersMade","statistics_threePointersMade"],
    "TPA": ["3PA","threePointersAttempted","statistics_threePointersAttempted"],
    "FTM": ["FTM","freeThrowsMade","statistics_freeThrowsMade"],
    "FTA": ["FTA","freeThrowsAttempted","statistics_freeThrowsAttempted"],
    "STL": ["STL","steals","statistics_steals"],
    "BLK": ["BLK","blocks","statistics_blocks"],
}

COLS: Dict[str, Optional[str]] = {k: pick_col(df, v) for k, v in CAND.items()}

# basic requirements
need = ["player_code","game_date","MIN","PTS","REB","AST","PIR"]
missing = [k for k in need if not COLS.get(k)]
if missing:
    st.error(f"Λείπουν βασικές στήλες: {missing}. Στείλε μου 1–2 γραμμές από το CSV για να ευθυγραμμίσουμε τα aliases.")
    st.stop()

# cast to proper types
df[COLS["game_date"]] = pd.to_datetime(df[COLS["game_date"]], errors="coerce")
for key in ["MIN","PTS","REB","AST","PIR","TOV","FDR","FGM","FGA","TPM","TPA","FTM","FTA","STL","BLK"]:
    c = COLS.get(key)
    if c and df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ==================== NEW INDICES ====================
eps = 1e-9
MIN = df[COLS["MIN"]].clip(lower=0.01)

# per-minute rates
def safe_rate(colkey: str) -> pd.Series:
    c = COLS.get(colkey)
    if not c:
        return pd.Series(np.nan, index=df.index)
    return df[c] / MIN

r_pts = safe_rate("PTS")
r_reb = safe_rate("REB")
r_ast = safe_rate("AST")
r_pir = safe_rate("PIR")
r_tov = safe_rate("TOV")
r_fdr = safe_rate("FDR")

def zscore(x: pd.Series) -> pd.Series:
    m = np.nanmean(x)
    s = np.nanstd(x) + eps
    return np.clip((x - m)/s, -3, 3)

z_MIN = zscore(MIN)
z_pts, z_reb, z_ast, z_pir, z_tov, z_fdr = map(zscore, [r_pts, r_reb, r_ast, r_pir, r_tov, r_fdr])

# Deep Impact Index (ανά παιχνίδι)
raw_dii = (
    0.25*z_pir + 0.22*z_pts + 0.20*z_reb + 0.20*z_ast + 0.10*z_fdr - 0.17*z_tov
    + 0.08*z_MIN
)

# Evenness bonus (να μην είναι μονοδιάστατο)
pos = np.column_stack([
    np.maximum(z_pts, 0),
    np.maximum(z_reb, 0),
    np.maximum(z_ast, 0),
    np.maximum(z_fdr, 0),
])
shares = pos / (pos.sum(axis=1, keepdims=True) + eps)
hhi = (shares**2).sum(axis=1)
evenness = (1 - hhi) / (1 - 1/4)  # scale to [0,1]
dii = raw_dii * (0.7 + 0.3*evenness)

# Stability & Form per player (rolling windows)
pcode = COLS["player_code"]
gdate = COLS["game_date"]
df = df.sort_values([pcode, gdate]).copy()

pir = df[COLS["PIR"]]
# rolling mean/std last 5 per player
pir_mean5 = pir.groupby(df[pcode]).transform(lambda s: s.rolling(5, min_periods=1).mean())
pir_std5  = pir.groupby(df[pcode]).transform(lambda s: s.rolling(5, min_periods=1).std())
cv5 = pir_std5 / (pir_mean5 + eps)
ss = (1 / (1 + cv5)) * 0.6 + pd.Series(evenness, index=df.index) * 0.4

# Form Momentum: avg last 3 − avg previous 3
pir_mean3 = pir.groupby(df[pcode]).transform(lambda s: s.rolling(3, min_periods=1).mean())
pir_prev3 = pir.shift(3).groupby(df[pcode]).transform(lambda s: s.rolling(3, min_periods=1).mean())
fm = (pir_mean3 - pir_prev3)

df["DII"] = pd.Series(dii, index=df.index).round(3)
df["SS"]  = ss.round(3)
df["FM"]  = fm.round(3)

# ==================== LIVE SEARCH + FILTERS ====================
name_col = COLS["player_name"] or pcode
st.text_input("🔎 Live search (player name/code)", value="", key="player_search")
query = st.session_state.get("player_search","").strip().lower()
fdf = df
if query:
    fdf = fdf[fdf[name_col].astype(str).str.lower().str.contains(query) | fdf[pcode].astype(str).str.lower().str.contains(query)]

# ==================== CLEAN TABLE (only relevant stats) ====================
wanted = [
    name_col, pcode, COLS["team_name"], gdate, COLS["home_away"], COLS["opponent"],
    COLS["MIN"], COLS["PTS"], COLS["REB"], COLS["AST"], COLS["PIR"], COLS["FDR"], COLS["TOV"],
    COLS["FGM"], COLS["FGA"], COLS["TPM"], COLS["TPA"], COLS["FTM"], COLS["FTA"], COLS["STL"], COLS["BLK"],
    "DII","SS","FM"
]
show_cols = [c for c in wanted if c and c in fdf.columns]
view = fdf[show_cols].copy()

# pretty headers
pretty = {
    name_col: "Player", pcode: "Code", COLS["team_name"]: "Team",
    gdate: "Date", COLS["home_away"]: "H/A", COLS["opponent"]: "Opp",
    COLS["MIN"]: "MIN", COLS["PTS"]: "PTS", COLS["REB"]: "REB", COLS["AST"]: "AST",
    COLS["PIR"]: "PIR", COLS["FDR"]: "FDR", COLS["TOV"]: "TOV",
    COLS["FGM"]: "FGM", COLS["FGA"]: "FGA", COLS["TPM"]: "3PM", COLS["TPA"]: "3PA",
    COLS["FTM"]: "FTM", COLS["FTA"]: "FTA", COLS["STL"]: "STL", COLS["BLK"]: "BLK",
    "DII": "Deep Impact", "SS": "Stability", "FM": "Form Δ(3vs3)"
}
view = view.rename(columns={k:v for k,v in pretty.items() if k in view.columns})

# ==================== SORTING ====================
sortable = ["PIR","PTS","REB","AST","MIN","FDR","TOV","DII","SS","FM","3PM","STL","BLK"]
# keep only those present
present = [lbl for lbl in sortable if (lbl in view.columns)]
sort_label = st.selectbox("Ταξινόμηση", present, index=present.index("PIR") if "PIR" in present else 0)
view = view.sort_values(by=sort_label, ascending=False)

# ==================== SHOW TABLE ====================
st.dataframe(view, use_container_width=True)
st.caption("DII: «βαθιά» επίδραση ανά λεπτό • SS: σταθερότητα (τελευταία 5) • FM: μομέντουμ φόρμας (3 vs προηγ. 3).")
