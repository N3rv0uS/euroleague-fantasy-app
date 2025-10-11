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
    return f'<a href="?{qs}" target="_blank" style="text-decoration:none;">{name}</a>'

show = df.copy()
show["player_name"] = [
    _plink(code, name) for code, name in zip(show["player_code"], show["player_name"])
]

cols = ["player_name", "player_team_name", "gamesPlayed", "minutesPlayed", "pir"]
cols = [c for c in cols if c in show.columns]

st.markdown(
    show[cols].rename(columns={"player_name": "Player"}).to_html(index=False, escape=False),
    unsafe_allow_html=True,
)

@st.cache_data(ttl=3600)
def scrape_gamelog_table(player_url: str) -> pd.DataFrame:
    import requests
    from bs4 import BeautifulSoup

    if not player_url:
        return None

    headers = {
        "User-Agent": "eurol-app/1.0",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "el,en;q=0.8",
    }

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

    s = requests.Session()

    for v in variants:
        try:
            r = s.get(v, headers=headers, timeout=20, allow_redirects=True)
            soup = BeautifulSoup(r.text, "lxml")
            tables = []
            for t in soup.find_all("table"):
                try:
                    part = pd.read_html(str(t))[0]
                    tables.append(part)
                except Exception:
                    continue
            if not tables:
                continue

            def score(df):
                cols = [re.sub(r"\W+", "", str(c).lower()) for c in df.columns]
                keys = ["pir", "min", "λεπ", "pts", "πον", "date", "ημ", "opponent", "αντιπ"]
                return sum(any(k in c for c in cols) for k in keys)

            return max(tables, key=score)
        except Exception:
            continue

    return None


def show_player_page(player_code: str):
    import pandas as pd

    pname = str(player_code)
    try:
        urls_df = pd.read_csv("out/player_urls_2025.csv")
        row = urls_df[urls_df["player_code"].astype(str) == str(player_code)].head(1)
    except Exception:
        row = pd.DataFrame()

    player_url = None
    if not row.empty:
        player_url = str(row.iloc[0].get("player_url", "")).strip()
        pname = row.iloc[0].get("Player", pname)

    st.title(f"{pname} — Αναλυτικά (Game-by-Game)")

    if not player_url:
        st.warning("Δεν βρέθηκε player_url για αυτόν τον παίκτη.")
        return

    gl = scrape_gamelog_table(player_url)

    if gl is None or gl.empty:
        st.warning("Δεν βρέθηκε πίνακας gamelogs στη σελίδα.")
        st.markdown(f"[Άνοιγμα επίσημου προφίλ]({player_url})")
        return

    st.dataframe(gl, use_container_width=True)
    for c in ["Πόντοι", "PTS", "PIR", "pir"]:
        if c in gl.columns:
            st.line_chart(gl[c])


pc = st.query_params.get("player_code")
if pc:
    show_player_page(pc)
    st.stop()
