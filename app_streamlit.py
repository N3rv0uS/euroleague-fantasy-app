# app_streamlit.py
import os
from pathlib import Path
from urllib.parse import urlencode
import re

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EuroLeague – Season Averages", layout="wide")

# -------------------- CONFIG --------------------
SEASON = os.getenv("SEASON", "2025")     # μπορείς να το αλλάξεις σε E2025 αν θέλεις
OUT_DIR = Path("out")

PLAYERS_CANDIDATES = [
    OUT_DIR / f"players_{SEASON}_perGame.csv",
]
# αν έρθει απλό "2025", δοκίμασε και σκέτα
if not (SEASON.startswith("E") or SEASON.startswith("U")):
    PLAYERS_CANDIDATES += [
        OUT_DIR / f"players_E{SEASON}_perGame.csv",
        OUT_DIR / f"players_U{SEASON}_perGame.csv",
    ]

PLAYER_URLS_PATH = OUT_DIR / f"player_urls_{SEASON}.csv"
GAMELOGS_CANDIDATES = [
    OUT_DIR / f"player_gamelogs_{SEASON}_perGame.csv",
]

# -------------------- HELPERS --------------------
def first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None

def pick_name_col(df: pd.DataFrame) -> str:
    for c in ["Player", "player_name", "name", "FullName", "full_name"]:
        if c in df.columns:
            return c
    # fallback: πρώτη στήλη που μοιάζει με όνομα
    for c in df.columns:
        if df[c].astype(str).str.contains(r"\s|,").any():
            return c
    return df.columns[0]

def pick_code_col(df: pd.DataFrame) -> str | None:
    for c in ["player_code", "code", "playerId", "playerID", "id"]:
        if c in df.columns:
            return c
    return None

# -------------------- LOAD DATA --------------------
players_path = first_existing(PLAYERS_CANDIDATES)
if not players_path:
    st.error("Δεν βρέθηκε το CSV των season averages στο `out/`.")
    st.stop()

df_players = pd.read_csv(players_path)

# Ενοποίηση ονόματος σε "Player"
name_col = pick_name_col(df_players)
if name_col != "Player":
    df_players = df_players.rename(columns={name_col: "Player"})
code_col = pick_code_col(df_players)

# Optional: προϋπάρχον gamelogs CSV
gamelogs_path = first_existing(GAMELOGS_CANDIDATES)
df_gamelogs = None
if gamelogs_path:
    try:
        df_gamelogs = pd.read_csv(gamelogs_path)
    except Exception:
        df_gamelogs = None

# -------------------- SCRAPER (SAFE) --------------------
@st.cache_data(ttl=3600)
def scrape_gamelog_table(player_url: str) -> pd.DataFrame | None:
    """Δοκιμάζει εναλλακτικές (EL/EN, με/χωρίς slash), διαβάζει ΜΟΝΟ <table> nodes.
    Επιστρέφει None αντί για raise όταν δεν βρεθούν πίνακες."""
    if not player_url:
        return None

    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        # Λείπει bs4 στο περιβάλλον
        return None

    headers = {
        "User-Agent": "eurol-app/1.2 (+stats)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "el,en;q=0.8",
    }

    # Παραλλαγές URL
    variants = []
    u = player_url.strip()
    variants.append(u)
    variants.append(u.rstrip("/"))
    if not u.endswith("/"):
        variants.append(u + "/")
    if "/el/" in u:
        variants.append(u.replace("/el/", "/en/"))
    if "/en/" in u:
        variants.append(u.replace("/en/", "/el/"))

    s = requests.Session()

    def _pick_best_table(html: str) -> pd.DataFrame | None:
        soup = BeautifulSoup(html, "lxml")
        tables = []
        for t in soup.find_all("table"):
            try:
                part = pd.read_html(str(t))[0]  # διαβάζουμε ΜΟΝΟ την εκάστοτε <table>
                tables.append(part)
            except Exception:
                continue
        if not tables:
            return None

        def score(df):
            cols = [re.sub(r"\W+", "", str(c).lower()) for c in df.columns]
            keys = ["pir", "min", "λεπ", "pts", "πον", "date", "ημ", "opponent", "αντιπ"]
            return sum(any(k in c for c in cols) for k in keys)

        return max(tables, key=score)

    for v in variants:
        try:
            r = s.get(v, headers=headers, timeout=20, allow_redirects=True)
            best = _pick_best_table(r.text)
            if best is not None and not best.empty:
                return best
        except Exception:
            continue

    return None

# -------------------- PLAYER PAGE --------------------
def show_player_page(player_code: str):
    """Δείχνει αναλυτικά ανά παίκτη.
    1) Αν υπάρχει gamelogs CSV → το χρησιμοποιεί.
    2) Αλλιώς προσπαθεί να βρει πίνακα από το official profile URL (PLAYER_URLS_PATH)."""
    pname = str(player_code)

    # 1) Προτίμησε τα τοπικά gamelogs (αν υπάρχουν)
    if df_gamelogs is not None and not df_gamelogs.empty:
        # βρες το κλειδί
        key = None
        for k in ["player_code", "code", "playerId", "id"]:
            if k in df_gamelogs.columns:
                key = k
                break
        if key:
            m = df_gamelogs[df_gamelogs[key].astype(str) == str(player_code)].copy()
            if not m.empty:
                # απόπειρα για όνομα
                for nc in ["Player", "player_name", "name"]:
                    if nc in m.columns and m[nc].notna().any():
                        pname = str(m[nc].dropna().iloc[0])
                        break

                st.title(f"{pname} — Αναλυτικά (Game-by-Game)")
                # sort by date αν υπάρχει
                for dc in ["game_date", "gameDate", "Date", "date"]:
                    if dc in m.columns:
                        with pd.option_context("mode.chained_assignment", None):
                            m[dc] = pd.to_datetime(m[dc], errors="coerce")
                            m.sort_values(dc, inplace=True)
                        break
                st.dataframe(m, use_container_width=True)
                for c in ["Πόντοι", "PTS", "PIR", "pir"]:
                    if c in m.columns:
                        st.line_chart(m.set_index(m.columns[0])[c] if m.columns.size else m[c])
                return  # ✅ τελειώσαμε

    # 2) Fallback: πάρε URL από mapping και κάνε scraping
    player_url = None
    if PLAYER_URLS_PATH.exists():
        try:
            urls_df = pd.read_csv(PLAYER_URLS_PATH)
            row = urls_df[urls_df["player_code"].astype(str) == str(player_code)].head(1)
            if not row.empty:
                player_url = str(row.iloc[0].get("player_url", "")).strip()
                pname = row.iloc[0].get("Player", pname)
        except Exception:
            player_url = None

    st.title(f"{pname} — Αναλυτικά (Game-by-Game)")

    if not player_url:
        st.warning("Δεν βρέθηκε σύνδεσμος προφίλ για τον παίκτη στο mapping CSV.")
        return

    gl = scrape_gamelog_table(player_url)
    if gl is None or gl.empty:
        st.warning("Δεν εντοπίστηκε HTML πίνακας gamelogs στη σελίδα του παίκτη.")
        st.markdown(f"[Άνοιγμα επίσημου προφίλ]({player_url})")
        return

    st.dataframe(gl, use_container_width=True)
    for c in ["Πόντοι", "PTS", "PIR", "pir"]:
        if c in gl.columns:
            st.line_chart(gl[c])

# -------------------- ROUTER (ΚΛΙΚ ΣΤΟ ΟΝΟΜΑ) --------------------
pc = st.query_params.get("player_code")
if pc:
    show_player_page(pc)
    st.stop()

# -------------------- MAIN: SEASON AVERAGES TABLE --------------------
st.title("Season Averages")

# CSS: λίγο μικρότερο font + άνετο padding
st.markdown(
    """
<style>
.small-table table { font-size: 13.5px; line-height: 1.25; }
.small-table th, .small-table td { padding: 6px 10px; }
.small-table a { color: inherit; text-decoration: none; }
</style>
""",
    unsafe_allow_html=True,
)

# Κάνε το Player clickable (χρειάζεται player_code)
if code_col:
    df_players = df_players.copy()
    df_players["Player"] = [
        f'<a href="?{urlencode({"player_code": str(code)})}">{name}</a>'
        for code, name in zip(df_players[code_col], df_players["Player"])
    ]

# Προτεραιότητες στήλες: βάλε όσες βρίσκουμε, οι υπόλοιπες ακολουθούν
pref = [
    "Player", "Team", "player_team_name",
    "GP", "gamesPlayed", "GS",
    "Min", "minutesPlayed",
    "PTS", "PIR",
    # shooting splits
    "2PM","2PA","2P%","3PM","3PA","3P%","FTM","FTA","FT%",
    # rebounds & playmaking & defense
    "OR","DR","TR","AST","STL","BLK","BLKA","TO","FC","FD",
    # advanced αν υπάρχουν
    "TS%","eFG%","FTR","Usage/min","PTS/min","TR/min","AST/min","FD/min","STL/min","BLK/min","Stocks/min","TO/min",
    "BCI","All_Score","PredictScore","Stability","Form3",
]
display_cols = [c for c in pref if c in df_players.columns] + [c for c in df_players.columns if c not in pref]

# Show more / less κάτω από τον πίνακα
is_all = st.session_state.get("show_all", False)
disp = df_players[display_cols].reset_index(drop=True)
if not is_all:
    disp = disp.head(30)

# Render
st.markdown(
    f"<div class='small-table'>{disp.to_html(index=False, escape=False)}</div>",
    unsafe_allow_html=True,
)

# Buttons κάτω από τον πίνακα
col_btn = st.container()
with col_btn:
    if not is_all:
        if st.button("Show more"):
            st.session_state["show_all"] = True
            st.rerun()
    else:
        if st.button("Show less"):
            st.session_state["show_all"] = False
            st.rerun()
