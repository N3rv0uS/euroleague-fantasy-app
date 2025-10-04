# app.py
import streamlit as st
import pandas as pd
import subprocess
import sys
from pathlib import Path

st.set_page_config(page_title="EuroLeague Fantasy", layout="wide")

OUT_DIR = Path("out")

st.title("EuroLeague Fantasy – Season & Game Logs")

mode_kind = st.radio("Dataset", ["Season averages", "Game logs"], horizontal=True)
season = st.text_input("Season", value="2025")
competition = st.selectbox("Competition", ["E", "U"], index=0)
stat_mode = st.selectbox("Mode", ["perGame"], index=0)

col_run, col_players = st.columns([2,3])
with col_players:
    players_in = st.text_input("Players (optional, comma-separated player codes)", value="")

with col_run:
    if st.button("Run now"):
        kind_arg = "season" if mode_kind == "Season averages" else "gamelogs"
        cmd = [
            sys.executable, "fetch_euroleague_stats.py",
            "--kind", kind_arg,
            "--seasons", season,
            "--competition", competition,
            "--mode", stat_mode,
            "--out", str(OUT_DIR),
        ]
        if players_in.strip() and kind_arg == "gamelogs":
            cmd += ["--players", players_in.strip()]
        with st.spinner("Fetching data..."):
            try:
                out = subprocess.check_output(cmd, text=True)
                st.success("Done.")
                st.code(out)
            except subprocess.CalledProcessError as e:
                st.error(f"Error running script: {e.output or e}")

# Load data
if mode_kind == "Season averages":
    f = OUT_DIR / f"players_{season}_{stat_mode}.csv"
else:
    f = OUT_DIR / f"player_gamelogs_{season}_{stat_mode}.csv"

if f.exists():
    df = pd.read_csv(f)

    st.subheader("Filters")
    if mode_kind == "Game logs":
        # Προσπάθεια να βρούμε standard columns
        date_cols = [c for c in df.columns if "date" in c.lower()]
        opponent_cols = [c for c in df.columns if "opponent" in c.lower() or "opposition" in c.lower() or c.lower().endswith("_opp")]

        # Player label columns
        player_label_cols = [c for c in df.columns if c.endswith("player_shortName") or c.endswith("player_displayName") or c.endswith("player_fullName")]
        player_code_col = "player_code" if "player_code" in df.columns else None

        # Δημιούργησε ετικέτα για player επιλογή
        if player_code_col:
            df["_player_label"] = df[player_code_col].astype(str)
            if player_label_cols:
                lbl = player_label_cols[0]
                df["_player_label"] = df["_player_label"] + " • " + df[lbl].astype(str)

            players_opts = sorted(df[player_code_col].astype(str).unique().tolist())
            chosen_player = st.selectbox("Player", players_opts)
            df = df[df[player_code_col].astype(str) == str(chosen_player)]

        # φίλτρα ημερομηνίας (αν έχουμε στήλη)
        if date_cols:
            dcol = date_cols[0]
            # robust parsing
            df["_gdate"] = pd.to_datetime(df[dcol], errors="coerce")
            min_d, max_d = df["_gdate"].min(), df["_gdate"].max()
            if pd.notnull(min_d) and pd.notnull(max_d):
                dr = st.date_input("Date range", value=(min_d.date(), max_d.date()))
                if isinstance(dr, tuple) and len(dr) == 2:
                    df = df[(df["_gdate"] >= pd.Timestamp(dr[0])) & (df["_gdate"] <= pd.Timestamp(dr[1]))]

        # αντίπαλος
        if opponent_cols:
            ocol = opponent_cols[0]
            opps = ["(All)"] + sorted([x for x in df[ocol].dropna().astype(str).unique().tolist()])
            chosen_opp = st.selectbox("Opponent", opps)
            if chosen_opp != "(All)":
                df = df[df[ocol].astype(str) == chosen_opp]

        # home/away (ψάχνουμε για πιθανές στήλες)
        ha_cols = [c for c in df.columns if "homeaway" in c.lower() or c.lower().endswith("_homeAway")]
        if ha_cols:
            hcol = ha_cols[0]
            has = ["(All)"] + sorted([x for x in df[hcol].dropna().astype(str).unique().tolist()])
            chosen_ha = st.selectbox("Home/Away", has)
            if chosen_ha != "(All)":
                df = df[df[hcol].astype(str) == chosen_ha]

        # chart (PTS ή PIR αν υπάρχουν)
        metric_candidates = [c for c in df.columns if c.upper() in ("PTS", "PIR")]
        if metric_candidates:
            metric = st.selectbox("Metric", metric_candidates)
            if "_gdate" in df.columns:
                chart_df = df.sort_values("_gdate")[["_gdate", metric]].rename(columns={"_gdate": "Date"})
                st.line_chart(chart_df, x="Date", y=metric, height=300)

    st.subheader("Table")
    st.dataframe(df, use_container_width=True)
else:
    st.info(f"Δεν βρέθηκε αρχείο: {f}")
