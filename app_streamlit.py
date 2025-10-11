# app_streamlit.py
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import os, re, pandas as pd, streamlit as st
from urllib.parse import urlencode

SEASON = "2025"  # ή E2025 αν έτσι δουλεύεις
avg_path = f"out/players_{SEASON}_perGame.csv"
urls_path = f"out/player_urls_{SEASON}.csv"

df_avg = pd.read_csv(avg_path)
df_urls = pd.read_csv(urls_path)
df = df_avg.merge(df_urls[["player_code", "player_url"]], on="player_code", how="left")

qp = st.query_params
player_code = qp.get("player_code")


@st.cache_data(ttl=3600)
def _plink(code, name):
    qs = urlencode({"player_code": str(code)})
    # target=_blank για νέο tab (βγάλε το αν δεν το θες)
    return f'<a href="?{qs}" target="_blank" style="text-decoration:none;">{name}</a>'


show = df.copy()
# Αν το δικό σου column είναι "Player" άλλαξέ το ανάλογα
show["player_name"] = [
    _plink(code, name) for code, name in zip(show["player_code"], show["player_name"])
]

cols = ["player_name", "player_team_name", "gamesPlayed", "minutesPlayed", "pir"]
cols = [c for c in cols if c in show.columns]

st.markdown(
    show[cols].rename(columns={"player_name": "Player"}).to_html(index=False, escape=False),
    unsafe_allow_html=True,
)


def scrape_gamelog_table(player_url: str) -> pd.DataFrame:
    import requests

    html = requests.get(player_url, headers={"User-Agent": "eurol-app/1.0"}).text
    tables = pd.read_html(html)

    def score(df):
        cols = [re.sub(r"\W+", "", str(c).lower()) for c in df.columns]
        keys = ["pir", "min", "λεπ", "pts", "πον", "date", "ημ", "opponent", "αντιπ"]
        return sum(any(k in c for c in cols) for k in keys)

    best = max(tables, key=score)
    return best


def _pick_best_table(html: str) -> Optional[pd.DataFrame]:
    try:
        tables = pd.read_html(html)
    except ValueError:
        tables = []

    if not tables:
        return None

    def score(df):
        cols = [re.sub(r"\W+", "", str(c).lower()) for c in df.columns]
        keys = ["pir", "min", "λεπ", "pts", "πον", "date", "ημ", "opponent", "αντιπ"]
        return sum(any(k in c for c in cols) for k in keys)

    best = max(tables, key=score)
    return best


def show_player_page(player_code: str):
    import pandas as pd
    import requests

    urls = pd.read_csv("out/player_urls_2025.csv")
    row = urls[urls["player_code"].astype(str) == str(player_code)].head(1)
    if row.empty or not str(row.iloc[0].get("player_url", "")).strip():
        st.error("Δεν βρέθηκε player_url για αυτόν τον παίκτη.")
        st.stop()

    player_url = row.iloc[0]["player_url"]
    player_name = row.iloc[0].get("Player", str(player_code))

    st.title(f"{player_name} — Αναλυτικά (Game-by-Game)")

    # Παίξε με παραλλαγές URL (slash & γλώσσα)
    variants = []
    u = (player_url or "").strip()
    if not u:
        return None
    variants.append(u)
    variants.append(u.rstrip("/"))
    if not u.endswith("/"):
        variants.append(u + "/")
    if "/el/" in u:
        variants.append(u.replace("/el/", "/en/"))
    if "/en/" in u:
        variants.append(u.replace("/en/", "/el/"))

    for v in variants:
        try:
            html = requests.get(v, headers={"User-Agent": "eurol-app/1.0"}, timeout=20).text
            df_best = _pick_best_table(html)
            if df_best is not None and not df_best.empty:
                st.dataframe(df_best, use_container_width=True)
                return
        except Exception:
            continue

    st.error("Δεν βρέθηκαν δεδομένα gamelog για αυτόν τον παίκτη.")


if player_code:
    try:
        show_player_page(player_code)
        st.stop()
    except Exception as e:
        st.error(f"Σφάλμα κατά την εμφάνιση του παίκτη: {e}")
