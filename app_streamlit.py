# app_streamlit.py
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
import streamlit as st

# ---------- ΡΥΘΜΙΣΕΙΣ ----------
OUT_DIR = Path("out")
st.set_page_config(page_title="EuroLeague Fantasy – Player Game Logs", layout="wide")


# ---------- ΒΟΗΘΗΤΙΚΑ ----------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    for kwargs in ({}, {"sep": ";"}, {"encoding": "utf-16"}):
        try:
            df = pd.read_csv(path, **kwargs)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:
            continue
    return None


def make_players_path(season: str, mode: str) -> Path:
    return OUT_DIR / f"players_{season}_{mode}.csv"


def make_gamelogs_path(season: str, mode: str) -> Path:
    return OUT_DIR / f"player_gamelogs_{season}_{mode}.csv"


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_players_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map aliases -> ζητούμενες στήλες & κράτα μόνο αυτές που έχουμε."""

    # --- Rename aliases σε canonical ονόματα που ζητάς ---
    rename_map: Dict[str, str] = {}

    # IDs / meta
    for a in ["player_code", "code", "playerCode"]:
        if a in df.columns: rename_map[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns: rename_map[a] = "Player"
    for a in ["team_code", "player_team_code", "teamCode", "team"]:
        if a in df.columns: rename_map[a] = "Team"
    for a in ["team_name", "player_team_name", "teamName"]:
        if a in df.columns and "Team" not in rename_map.values(): rename_map[a] = "Team"

    # Games
    for a in ["gamesPlayed", "GP", "games", "G"]:
        if a in df.columns: rename_map[a] = "GP"
    for a in ["gamesStarted", "GS"]:
        if a in df.columns: rename_map[a] = "GS"

    # Minutes / PIR
    for a in ["minutesPlayed", "MIN", "Minutes"]:
        if a in df.columns: rename_map[a] = "Min"
    for a in ["pir", "PIR", "EFF", "efficiency"]:
        if a in df.columns: rename_map[a] = "PIR"

    # Shooting splits
    # 2P
    for a in ["twoPointersMade", "2PM"]:
        if a in df.columns: rename_map[a] = "2PM"
    for a in ["twoPointersAttempted", "2PA"]:
        if a in df.columns: rename_map[a] = "2PA"
    for a in ["twoPointersPercentage", "2P%"]:
        if a in df.columns: rename_map[a] = "2P%"

    # 3P
    for a in ["threePointersMade", "3PM"]:
        if a in df.columns: rename_map[a] = "3PM"
    for a in ["threePointersAttempted", "3PA"]:
        if a in df.columns: rename_map[a] = "3PA"
    for a in ["threePointersPercentage", "3P%"]:
        if a in df.columns: rename_map[a] = "3P%"

    # FT
    for a in ["freeThrowsMade", "FTM"]:
        if a in df.columns: rename_map[a] = "FTM"
    for a in ["freeThrowsAttempted", "FTA"]:
        if a in df.columns: rename_map[a] = "FTA"
    for a in ["freeThrowsPercentage", "FT%"]:
        if a in df.columns: rename_map[a] = "FT%"

    # Rebounds
    for a in ["offensiveRebounds", "OR", "OREB"]:
        if a in df.columns: rename_map[a] = "OR"
    for a in ["defensiveRebounds", "DR", "DREB"]:
        if a in df.columns: rename_map[a] = "DR"
    for a in ["totalRebounds", "REB", "TR", "TRB"]:
        if a in df.columns: rename_map[a] = "TR"

    # Points / playmaking / defense / fouls
    for a in ["pointsScored", "PTS", "Points"]:
        if a in df.columns: rename_map[a] = "PTS"
    for a in ["assists", "AST"]:
        if a in df.columns: rename_map[a] = "AST"
    for a in ["steals", "STL"]:
        if a in df.columns: rename_map[a] = "STL"
    for a in ["turnovers", "TOV", "TO"]:
        if a in df.columns: rename_map[a] = "TO"
    for a in ["blocks", "BLK"]:
        if a in df.columns: rename_map[a] = "BLK"
    for a in ["blocksAgainst", "BLKA", "BLK_AG"]:
        if a in df.columns: rename_map[a] = "BLKA"
    for a in ["foulsCommited", "foulsCommitted", "FC", "FLS_CM"]:
        if a in df.columns: rename_map[a] = "FC"
    for a in ["foulsDrawn", "FD", "FLS_RV"]:
        if a in df.columns: rename_map[a] = "FD"

    # Season / competition (τα κρατάμε κρυφά για join)
    for a in ["season", "Season"]:
        if a in df.columns: rename_map[a] = "season"
    for a in ["competition", "Competition"]:
        if a in df.columns: rename_map[a] = "competition"

    df = df.rename(columns=rename_map)

    # Γέμισε ελάχιστα πεδία αν λείπουν
    for c in ["Player", "player_code", "Team"]:
        if c not in df.columns: df[c] = None

    # Υπολόγισε TR αν λείπει αλλά έχουμε OR+DR
    if "TR" not in df.columns and all(c in df.columns for c in ["OR", "DR"]):
        df["TR"] = df["OR"].fillna(0) + df["DR"].fillna(0)

    # Ζητούμενη σειρά στηλών
    target_cols = [
        "Player", "Team",
        "GP", "GS", "Min", "PTS",
        "2PM", "2PA", "2P%",
        "3PM", "3PA", "3P%",
        "FTM", "FTA", "FT%",
        "OR", "DR", "TR",
        "AST", "STL", "TO", "BLK", "BLKA",
        "FC", "FD", "PIR",
        # advanced (θα προστεθούν αργότερα):
        "BCI", "Stability", "Form3",
    ]

    # Κράτα μόνο ό,τι υπάρχει (θα προστεθούν advanced μετά)
    keep = [c for c in target_cols if c in df.columns and c not in ["BCI", "Stability", "Form3"]]
    # ταξινόμηση by PIR αν υπάρχει
    if "PIR" in df.columns:
        df = df.sort_values("PIR", ascending=False, na_position="last")

    return df, keep, target_cols


def normalize_gamelogs_df(df: pd.DataFrame) -> pd.DataFrame:
    # βασικά aliases
    ren = {}
    for a in ["player_code", "code", "playerCode"]:
        if a in df.columns: ren[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns: ren[a] = "Player"
    for a in ["team_code", "teamCode", "Team"]:
        if a in df.columns: ren[a] = "Team"
    for a in ["opponent", "opponent_code", "Opp", "Opponent"]:
        if a in df.columns: ren[a] = "opponent"
    for a in ["game_date", "date", "gameDate", "Date"]:
        if a in df.columns: ren[a] = "game_date"

    df = df.rename(columns=ren)

    # Parse ημερομηνίας
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Stat aliases -> canonical για gamelogs
    def _alias(dst, *alts):
        if dst not in df.columns:
            for a in alts:
                if a in df.columns:
                    df.rename(columns={a: dst}, inplace=True)
                    break

    _alias("MIN", "minutesPlayed", "Min", "Minutes")
    _alias("PTS", "pointsScored", "Points")
    _alias("TR", "totalRebounds", "REB", "TRB")
    _alias("AST", "assists")
    _alias("STL", "steals")
    _alias("TO", "turnovers", "TOV")
    _alias("BLK", "blocks")
    _alias("FC", "foulsCommitted", "foulsCommited", "FLS_CM")
    _alias("FD", "foulsDrawn", "FLS_RV", "FD")
    _alias("PIR", "pir", "EFF", "efficiency")

    return df


def compute_advanced(players_df: pd.DataFrame, gl_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Υπολογίζει BCI, Stability, Form3 και τα κάνει join στο players_df."""

    adv = pd.DataFrame(index=players_df.index)

    # --- BCI: Balanced Contribution Index ---
    # per-minute «γεμάτη» συνεισφορά: PTS, TR, AST, FD, PIR (scaled), με bonus για availability (Min)
    # ορισμοί per-minute (ασφαλής διαίρεση)
    def pm(x, y):
        x = x.fillna(0)
        y = y.replace(0, np.nan)
        return (x / y).fillna(0)

    PTS_pm = pm(players_df.get("PTS", 0), players_df.get("Min", 0))
    TR_pm  = pm(players_df.get("TR", 0),  players_df.get("Min", 0))
    AST_pm = pm(players_df.get("AST", 0), players_df.get("Min", 0))
    FD_pm  = pm(players_df.get("FD", 0),  players_df.get("Min", 0))
    PIR_pm = pm(players_df.get("PIR", 0), players_df.get("Min", 0))

    # γραμμικός συνδυασμός (weights), κλιμάκωση στο 0–100
    raw_bci = 0.35*PTS_pm + 0.25*TR_pm + 0.25*AST_pm + 0.10*FD_pm + 0.05*PIR_pm
    # bonus για πολλά λεπτά (Min / 30 cap στο 1.2)
    min_bonus = (players_df.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)
    raw_bci = raw_bci * min_bonus

    # scale 0–100 ανά dataset
    if raw_bci.max() > 0:
        BCI = 100 * (raw_bci / raw_bci.max())
    else:
        BCI = raw_bci
    adv["BCI"] = BCI.round(1)

    # --- Stability: 1/(1+CV) του PIR σε last 6 games (0–100) ---
    stability_map = {}
    form3_map = {}
    if gl_df is not None and not gl_df.empty:
        # δουλεύουμε per player_code αν υπάρχει, αλλιώς per Player
        key = "player_code" if "player_code" in gl_df.columns else "Player"
        if key not in players_df.columns:
            # αν δεν υπάρχει το ίδιο key στους players, προσπάθησε να χτίσεις map
            # με βάση Player name
            if key == "Player" and "Player" in players_df.columns:
                pass  # ok
            else:
                key = None

        if key is not None:
            # groupby παίκτη
            for pid, g in gl_df.groupby(key):
                g = g.sort_values("game_date")
                pir = g.get("PIR")
                if pir is None or pir.dropna().empty:
                    continue
                # Stability σε last 6
                lastN = pir.dropna().tail(6)
                if len(lastN) >= 3 and lastN.mean() != 0:
                    cv = lastN.std(ddof=0) / abs(lastN.mean())
                    stab = (1.0 / (1.0 + cv)) * 100.0
                    stability_map[pid] = float(np.clip(stab, 0, 100))
                # Form3: μέσο PIR τελευταίων 3
                last3 = pir.dropna().tail(3)
                if len(last3) > 0:
                    form3_map[pid] = float(last3.mean())

            # Ανάθεση πίσω στους players
            if "player_code" in players_df.columns and key == "player_code":
                adv["Stability"] = players_df["player_code"].map(stability_map)
                adv["Form3"] = players_df["player_code"].map(form3_map)
            elif "Player" in players_df.columns and key == "Player":
                adv["Stability"] = players_df["Player"].map(stability_map)
                adv["Form3"] = players_df["Player"].map(form3_map)

    # round
    if "Stability" in adv.columns:
        adv["Stability"] = adv["Stability"].round(1)
    if "Form3" in adv.columns:
        adv["Form3"] = adv["Form3"].round(1)

    # join
    out = players_df.join(adv)
    return out


def filter_players(df: pd.DataFrame, q: str, team: str, min_gp: int) -> pd.DataFrame:
    res = df.copy()
    if q:
        qlow = q.lower().strip()
        res = res[
            res.get("Player", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().str.contains(qlow)
            | res.get("player_code", pd.Series(index=res.index, dtype=str)).fillna("").astype(str).str.contains(qlow)
            | res.get("Team", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().str.contains(qlow)
        ]
    if team and team != "(Όλες)":
        res = res[res.get("Team", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().eq(team.lower())]
    if "GP" in res.columns and min_gp > 0:
        res = res[res["GP"].fillna(0) >= min_gp]
    return res


# ---------- UI ----------
st.title("EuroLeague Fantasy – Player Game Logs")

# Επιλογές header
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    season = st.selectbox("Season", ["2025"], index=0)
with c2:
    competition = st.selectbox("Competition", ["E", "U"], index=0)
with c3:
    mode = st.selectbox("Mode", ["perGame"], index=0)

players_path = make_players_path(season, mode)
gamelogs_path = make_gamelogs_path(season, mode)

players_df_raw = load_csv(players_path)
gamelogs_df_raw = load_csv(gamelogs_path)

st.caption(
    f"📄 Averages: `{players_path}` — "
    f"{'OK' if players_df_raw is not None else 'ΔΕΝ ΒΡΕΘΗΚΕ'}  |  "
    f"📄 Gamelogs: `{gamelogs_path}` — "
    f"{'OK' if gamelogs_df_raw is not None else 'ΔΕΝ ΒΡΕΘΗΚΕ'}"
)

if players_df_raw is None:
    st.error(f"Δεν βρέθηκε το αρχείο season averages: `{players_path}`.")
    st.stop()

# Normalize + keep requested columns
players_df_norm, base_cols, target_cols = normalize_players_df(players_df_raw)

# Normalize gamelogs (αν υπάρχουν) και υπολόγισε Advanced
gamelogs_df = None
if gamelogs_df_raw is not None and not gamelogs_df_raw.empty:
    gamelogs_df = normalize_gamelogs_df(gamelogs_df_raw.copy())

players_df = compute_advanced(players_df_norm, gamelogs_df)

# Ετοίμασε τελικές στήλες με προτεραιότητα στο ζητούμενο order
final_cols = []
for c in target_cols:
    if c in players_df.columns:
        final_cols.append(c)

# Φίλτρα / αναζήτηση
teams_list = ["(Όλες)"] + sorted(players_df.get("Team", pd.Series(dtype=str)).dropna().astype(str).unique(), key=lambda x: x.lower())
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    q = st.text_input("🔎 Live search (όνομα/κωδικός/ομάδα)", "")
with f2:
    team_sel = st.selectbox("Ομάδα", teams_list, index=0)
with f3:
    min_gp = st.number_input("Min GP", min_value=0, max_value=50, value=0, step=1)

filtered_players = filter_players(players_df, q, team_sel, min_gp)

st.subheader("Season Averages (με τις ζητούμενες στήλες + Advanced)")
st.dataframe(
    filtered_players[final_cols].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

# Επιλογή παίκτη και εμφάνιση gamelogs
left, right = st.columns([1, 2])
with left:
    st.markdown("### Επιλογή παίκτη")
    options = (
        filtered_players[["Player", "player_code"]]
        .dropna(subset=["Player"])
        .drop_duplicates()
        .sort_values("Player")
        .assign(label=lambda d: d["Player"] + "  (" + d["player_code"].astype(str) + ")")
    )
    if len(options) == 0:
        selected_label = None
        st.info("Κανένα αποτέλεσμα με τα τρέχοντα φίλτρα.")
    else:
        selected_label = st.selectbox(
            "Διάλεξε παίκτη",
            options["label"].tolist(),
            index=0,
            key="player_select",
        )
        selected_player_code = None
        if selected_label:
            sel_row = options[options["label"] == selected_label].iloc[0]
            selected_player_code = str(sel_row["player_code"])

with right:
    st.markdown("### Αναλυτικά (Game-by-Game)")
    if gamelogs_df is None or gamelogs_df.empty:
        st.warning("Δεν υπάρχουν gamelogs στο repository για να εμφανιστούν αναλυτικά.")
    else:
        player_gl = pd.DataFrame()
        if selected_label:
            if "player_code" in gamelogs_df.columns and selected_player_code is not None:
                player_gl = gamelogs_df[gamelogs_df["player_code"].astype(str) == selected_player_code]
            if player_gl.empty and "Player" in gamelogs_df.columns:
                # fall back σε όνομα
                p_name = sel_row["Player"]
                player_gl = gamelogs_df[gamelogs_df["Player"].astype(str).str.lower() == str(p_name).lower()]

        if selected_label and not player_gl.empty:
            csum1, csum2, csum3 = st.columns(3)
            with csum1: st.metric("Games", len(player_gl))
            with csum2: st.metric("PTS (avg)", round(player_gl.get("PTS", pd.Series([0])).mean(), 2))
            with csum3: st.metric("PIR (avg)", round(player_gl.get("PIR", pd.Series([0])).mean(), 2))

            # charts
            if "game_date" in player_gl.columns:
                if "PTS" in player_gl.columns:
                    sub = player_gl[["game_date", "PTS"]].dropna().sort_values("game_date").set_index("game_date")
                    st.line_chart(sub, height=180, use_container_width=True)
                if "PIR" in player_gl.columns:
                    sub = player_gl[["game_date", "PIR"]].dropna().sort_values("game_date").set_index("game_date")
                    st.line_chart(sub, height=180, use_container_width=True)

            # πίνακας gamelogs
            gl_cols_pref = ["game_date", "Team", "opponent", "MIN", "PTS", "TR", "AST", "STL", "TO", "BLK", "FC", "FD", "PIR"]
            gl_cols = [c for c in gl_cols_pref if c in player_gl.columns]
            extra = []
            for c in ["2FG", "3FG", "FT", "2FG_M", "2FG_A", "3FG_M", "3FG_A", "FT_M", "FT_A"]:
                if c in player_gl.columns: extra.append(c)
            gl_cols = gl_cols + extra

            st.dataframe(player_gl[gl_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            if selected_label:
                st.info("Δεν βρέθηκαν gamelogs για τον συγκεκριμένο παίκτη (ή το αρχείο είναι κενό).")

st.divider()
st.caption(
    "💡 Ο πίνακας Season Averages ακολουθεί ακριβώς τη δομή: "
    "Player, Team, GP, GS, Min, PTS, 2PM/2PA/2P%, 3PM/3PA/3P%, FTM/FTA/FT%, "
    "OR/DR/TR, AST, STL, TO, BLK, BLKA, FC, FD, PIR, καθώς και τα Advanced: BCI, Stability, Form3. "
    "Τα Stability & Form3 εμφανίζονται όταν υπάρχουν gamelogs."
)
