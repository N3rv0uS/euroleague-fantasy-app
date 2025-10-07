# app_streamlit.py
import os
import re
import pandas as pd
import streamlit as st
from urllib.parse import urlencode

st.set_page_config(page_title="Season Averages", layout="wide")

# ---------- CONFIG ----------
SEASON = os.getenv("SEASON", "2025")  # βάλε "E2025" αν έτσι αποθηκεύεις
AVG_CANDIDATES = [
    f"out/players_{SEASON}_perGame.csv",
    f"out/players_{SEASON.lstrip('EU')}_perGame.csv",
    f"out/players_E{SEASON}_perGame.csv" if not SEASON.startswith(("E","U")) else "",
]
URLS_PATH = f"out/player_urls_{SEASON}.csv"  # παράγεται από build_player_urls.py

# ---------- HELPERS ----------
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _pick_name_col(df: pd.DataFrame) -> str:
    for c in ["player_name", "Player", "name", "FullName", "full_name"]:
        if c in df.columns:
            return c
    # fallback: πάρε την πρώτη στήλη που μοιάζει με όνομα (έχει κόμμα ή κενά)
    for c in df.columns:
        if df[c].astype(str).str.contains(r"\s|,").any():
            return c
    return df.columns[0]

@st.cache_data(ttl=3600)
def scrape_gamelog_table(player_url: str) -> pd.DataFrame:
    import requests
    html = requests.get(player_url, headers={"User-Agent": "eurol-app/1.0"}, timeout=20).text
    tables = pd.read_html(html)
    def score(df):
        cols = [re.sub(r"\W+", "", str(c).lower()) for c in df.columns]
        keys = ["pir","min","λεπ","pts","πον","date","ημ","opponent","αντιπ"]
        return sum(any(k in c for c in cols) for k in keys)
    return max(tables, key=score)

def _player_link(code, text) -> str:
    qs = urlencode({"player_code": str(code)})
    # θες νέο tab; πρόσθεσε target="_blank"
    return f'<a href="?{qs}" style="text-decoration:none;">{text}</a>'

# ---------- LOAD DATA ----------
avg_path = _first_existing(AVG_CANDIDATES)
if not avg_path:
    st.error("Δεν βρέθηκε το CSV των season averages στο out/.")
    st.stop()

df_avg = pd.read_csv(avg_path)

# optional: merge με player_urls
df_urls = pd.read_csv(URLS_PATH) if os.path.exists(URLS_PATH) else pd.DataFrame(columns=["player_code","player_url"])
if "player_code" in df_avg.columns and not df_urls.empty:
    df_avg = df_avg.merge(df_urls[["player_code","player_url"]], on="player_code", how="left")
else:
    if "player_url" not in df_avg.columns:
        df_avg["player_url"] = None  # placeholder

# ---------- ROUTER ----------
qp = st.query_params
player_code = qp.get("player_code")

# ---------- PAGE: PLAYER DETAIL ----------
if player_code:
    # εντοπισμός γραμμής παίκτη
    mask = df_avg.get("player_code", df_avg.iloc[:,0]).astype(str) == str(player_code)
    row = df_avg[mask].head(1)
    if row.empty:
        st.error("Player not found.")
        st.stop()

    name_col = _pick_name_col(df_avg)
    name = row.iloc[0][name_col]
    url  = row.iloc[0].get("player_url")

    st.title(f"{name} — Αναλυτικά (Game-by-Game)")
    st.markdown(f"[⬅ Επιστροφή](/)", unsafe_allow_html=True)

    if not url or pd.isna(url):
        st.warning("Δεν υπάρχει αποθηκευμένο player_url για scraping. Φτιάξε/ενημέρωσε το out/player_urls_*.csv.")
        st.stop()

    try:
        gl = scrape_gamelog_table(url)
    except Exception as e:
        st.error(f"Αποτυχία ανάκτησης gamelog: {e}")
        st.stop()

    st.dataframe(gl, use_container_width=True)
    # προαιρετικά μικρά charts αν υπάρχουν οι στήλες
    for c in ["Πόντοι","PTS","PIR","pir"]:
        if c in gl.columns:
            st.line_chart(gl[c])
    st.stop()

# ---------- PAGE: MAIN (SEASON AVERAGES TABLE) ----------
st.title("Season Averages")

# κρατάμε ΟΛΕΣ τις στήλες, απλώς κάνουμε την 1η (όνομα) link
df_show = df_avg.copy()

name_col = _pick_name_col(df_show)
code_col = "player_code" if "player_code" in df_show.columns else None

if code_col:
    df_show[name_col] = [
        _player_link(code, name) for code, name in zip(df_show[code_col], df_show[name_col])
    ]
else:
    st.warning("Δεν βρέθηκε στήλη player_code — τα links θα είναι ανενεργά.")

# εμφανίζουμε όλες τις στήλες, με το όνομα πρώτο
cols = [name_col] + [c for c in df_show.columns if c != name_col]
html = df_show[cols].rename(columns={name_col: "Player"}).to_html(index=False, escape=False)
st.markdown(html, unsafe_allow_html=True)
