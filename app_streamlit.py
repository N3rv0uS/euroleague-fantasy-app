# app_streamlit.py
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# ---------- ΡΥΘΜΙΣΕΙΣ ----------
OUT_DIR = Path("out")  # φάκελος με τα CSV
st.set_page_config(page_title="EuroLeague Fantasy – Player Game Logs", layout="wide")

# ---------- ΒΟΗΘΗΤΙΚΑ ----------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        # δοκίμασε και UTF-16/semicolon αν χρειαστεί
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            return None
    # καθάρισμα whitespaces στα ονόματα στηλών
    df.columns = [c.strip() for c in df.columns]
    return df

def make_players_path(season: str, mode: str) -> Path:
    return OUT_DIR / f"players_{season}_{mode}.csv"

def make_gamelogs_path(season: str, mode: str) -> Path:
    return OUT_DIR / f"player_gamelogs_{season}_{mode}.csv"

def normalize_players_df(df: pd.DataFrame) -> pd.DataFrame:
    """Φέρνει τα βασικά aliases σε κοινά ονόματα πεδίων."""
    # πιθανά aliases
    rename_map = {}
    for a in ["player_code", "code", "playerCode"]:
        if a in df.columns: rename_map[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns: rename_map[a] = "player_name"
    for a in ["team_code", "player_team_code", "teamCode", "team"]:
        if a in df.columns: rename_map[a] = "team_code"
    for a in ["team_name", "player_team_name", "teamName"]:
        if a in df.columns: rename_map[a] = "team_name"
    for a in ["season", "Season"]:
        if a in df.columns: rename_map[a] = "season"
    for a in ["competition", "Competition"]:
        if a in df.columns: rename_map[a] = "competition"

    df = df.rename(columns=rename_map)

    # Στήλες που συνήθως θέλουμε να δείχνουμε στον πίνακα
    preferred_order = [
        "player_name", "player_code", "team_code", "team_name",
        "gamesPlayed", "minutesPlayed", "pointsScored",
        "totalRebounds", "assists", "steals", "turnovers", "blocks",
        "foulsDrawn", "pir",
    ]
    # Συχνά aliases stats
    alt = {
        "gamesPlayed": ["GP", "games", "G"],
        "minutesPlayed": ["MIN", "Minutes"],
        "pointsScored": ["PTS", "Points"],
        "totalRebounds": ["REB", "Reb", "TRB"],
        "assists": ["AST", "Assists"],
        "steals": ["STL"],
        "turnovers": ["TOV"],
        "blocks": ["BLK"],
        "foulsDrawn": ["FLS_RV", "FD", "FDR"],
        "pir": ["PIR", "EFF", "efficiency"],
    }
    for canonical, alts in alt.items():
        if canonical not in df.columns:
            for a in alts:
                if a in df.columns:
                    df = df.rename(columns={a: canonical})
                    break

    # Βεβαιώσου ότι υπάρχουν οι βασικές στήλες έστω κενές
    for c in ["player_name", "player_code", "team_code", "team_name"]:
        if c not in df.columns:
            df[c] = None

    # Ταξινόμηση
    if "pir" in df.columns:
        df = df.sort_values("pir", ascending=False, na_position="last")

    # Επιλογή εμφάνισης: μόνο στήλες που υπάρχουν
    show_cols = [c for c in preferred_order if c in df.columns]
    # πρόσθεσε χρήσιμες αν υπάρχουν
    for c in ["twoPointersPercentage", "threePointersPercentage", "freeThrowsPercentage"]:
        if c in df.columns and c not in show_cols:
            show_cols.append(c)
    # πάντα κράτα τα βασικά meta στο τέλος
    for c in ["season", "competition", "mode"]:
        if c in df.columns and c not in show_cols:
            show_cols.append(c)

    return df, show_cols

def normalize_gamelogs_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for a in ["player_code", "code", "playerCode"]:
        if a in df.columns: rename_map[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns: rename_map[a] = "player_name"
    for a in ["team_code", "teamCode", "Team"]:
        if a in df.columns: rename_map[a] = "team_code"
    for a in ["opponent", "opponent_code", "Opp", "Opponent"]:
        if a in df.columns: rename_map[a] = "opponent"
    for a in ["game_date", "date", "gameDate", "Date"]:
        if a in df.columns: rename_map[a] = "game_date"

    df = df.rename(columns=rename_map)

    # Parse ημερομηνίας
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Stats aliases
    alt = {
        "MIN": ["minutesPlayed", "Min", "Minutes"],
        "PTS": ["pointsScored", "Points"],
        "REB": ["totalRebounds", "Reb", "TRB"],
        "AST": ["assists"],
        "STL": ["steals"],
        "TOV": ["turnovers"],
        "BLK": ["blocks"],
        "FLS_CM": ["foulsCommited", "foulsCommitted", "PF"],
        "FLS_RV": ["foulsDrawn", "FD"],
        "PIR": ["pir", "EFF", "efficiency"],
    }
    for canonical, alts in alt.items():
        if canonical not in df.columns:
            for a in alts:
                if a in df.columns:
                    df = df.rename(columns={a: canonical})
                    break

    # Ταξινόμηση
    if "game_date" in df.columns:
        df = df.sort_values("game_date", ascending=False, na_position="last")

    return df

def filter_players(df: pd.DataFrame, q: str, team: str, min_gp: int) -> pd.DataFrame:
    res = df.copy()
    if q:
        qlow = q.lower().strip()
        res = res[
            res["player_name"].fillna("").str.lower().str.contains(qlow)
            | res["player_code"].fillna("").astype(str).str.contains(qlow)
            | res.get("team_name", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().str.contains(qlow)
            | res.get("team_code", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().str.contains(qlow)
        ]
    if team and team != "(Όλες)":
        teamlow = team.lower()
        mask = (
            res.get("team_code", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().eq(teamlow)
            | res.get("team_name", pd.Series(index=res.index, dtype=str)).fillna("").str.lower().eq(teamlow)
        )
        res = res[mask]
    if "gamesPlayed" in res.columns and min_gp > 0:
        res = res[res["gamesPlayed"].fillna(0) >= min_gp]
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

# Ενημερωτική μπάρα για τα αρχεία
st.caption(
    f"📄 Averages: `{players_path}` — "
    f"{'OK' if players_df_raw is not None else 'ΔΕΝ ΒΡΕΘΗΚΕ'}  |  "
    f"📄 Gamelogs: `{gamelogs_path}` — "
    f"{'OK' if gamelogs_df_raw is not None else 'ΔΕΝ ΒΡΕΘΗΚΕ'}"
)

if players_df_raw is None:
    st.error(f"Δεν βρέθηκε το αρχείο season averages: `{players_path}`. Τρέξε πρώτα το step για averages.")
    st.stop()

# Κανονικοποίηση
players_df, players_show_cols = normalize_players_df(players_df_raw)
teams_list = ["(Όλες)"] + sorted(
    pd.unique(
        players_df.get("team_name", pd.Series(dtype=str)).dropna().astype(str).tolist()
        + players_df.get("team_code", pd.Series(dtype=str)).dropna().astype(str).tolist()
    ),
    key=lambda x: x.lower()
)

# Φίλτρα / αναζήτηση
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    q = st.text_input("🔎 Live search (όνομα/κωδικός/ομάδα)", "")
with f2:
    team_sel = st.selectbox("Ομάδα", teams_list, index=0)
with f3:
    min_gp = st.number_input("Min GP", min_value=0, max_value=50, value=0, step=1)

filtered_players = filter_players(players_df, q, team_sel, min_gp)

st.subheader("Season Averages (πίνακας παικτών)")
st.dataframe(
    filtered_players[players_show_cols].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

# Επιλογή παίκτη (από τα φιλτραρισμένα αποτελέσματα)
left, right = st.columns([1, 2])
with left:
    st.markdown("### Επιλογή παίκτη")
    # Επιλογή με βάση όνομα αλλά κρατάμε και τον code
    options = (
        filtered_players[["player_name", "player_code"]]
        .dropna(subset=["player_name"])
        .drop_duplicates()
        .sort_values("player_name")
        .assign(label=lambda d: d["player_name"] + "  (" + d["player_code"].astype(str) + ")")
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
        # εξαγωγή code από το label
        if selected_label:
            sel_row = options[options["label"] == selected_label].iloc[0]
            selected_player_code = str(sel_row["player_code"])
        else:
            selected_player_code = None

with right:
    st.markdown("### Αναλυτικά (Game-by-Game)")
    if gamelogs_df_raw is None:
        st.warning(
            f"Δεν υπάρχει ακόμη αρχείο gamelogs για να εμφανιστούν αναλυτικά.\n\n"
            f"Περίμενε/τρέξε το update ώστε να δημιουργηθεί: `{gamelogs_path}`."
        )
    else:
        gamelogs_df = normalize_gamelogs_df(gamelogs_df_raw.copy())
        # Προσπάθησε να φιλτράρεις με βάση player_code — αν λείπει, fall back σε όνομα
        player_gl = pd.DataFrame()
        if selected_label:
            if "player_code" in gamelogs_df.columns:
                player_gl = gamelogs_df[gamelogs_df["player_code"].astype(str) == selected_player_code]
            if player_gl.empty and "player_name" in gamelogs_df.columns:
                # πάρε το όνομα από τα players
                p_name = sel_row["player_name"]
                player_gl = gamelogs_df[gamelogs_df["player_name"].astype(str).str.lower() == str(p_name).lower()]

        if selected_label and not player_gl.empty:
            # Μικρή σύνοψη
            csum1, csum2, csum3 = st.columns(3)
            with csum1:
                st.metric("Games", len(player_gl))
            with csum2:
                st.metric("PTS (avg)", round(player_gl.get("PTS", pd.Series([0])).mean(), 2))
            with csum3:
                st.metric("PIR (avg)", round(player_gl.get("PIR", pd.Series([0])).mean(), 2))

            # Γραφήματα
            chart_cols = []
            if "game_date" in player_gl.columns and "PTS" in player_gl.columns:
                chart_cols.append(("PTS", "Πόντοι"))
            if "game_date" in player_gl.columns and "PIR" in player_gl.columns:
                chart_cols.append(("PIR", "PIR"))
            if chart_cols:
                st.caption("📈 Εξέλιξη στα τελευταία παιχνίδια")
                for c, label in chart_cols:
                    sub = player_gl[["game_date", c]].dropna().sort_values("game_date")
                    sub = sub.set_index("game_date")
                    st.line_chart(sub, height=180, use_container_width=True)

            # Πίνακας αγώνα-αγώνα
            show_cols = []
            for c in ["game_date", "team_code", "opponent", "W", "L", "MIN", "PTS", "REB", "AST", "STL", "TOV", "BLK", "FLS_CM", "FLS_RV", "PIR"]:
                if c in player_gl.columns:
                    show_cols.append(c)
            extra = []
            for c in ["2FG", "3FG", "FT", "2FG_M", "2FG_A", "3FG_M", "3FG_A", "FT_M", "FT_A"]:
                if c in player_gl.columns: extra.append(c)
            show_cols = [*show_cols, *extra]

            st.dataframe(
                player_gl[show_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        else:
            if selected_label:
                st.info("Δεν βρέθηκαν gamelogs για τον συγκεκριμένο παίκτη (ή το αρχείο είναι κενό).")

# Υπόμνημα/βοήθεια
st.divider()
st.caption(
    "💡 Η σελίδα φορτώνει **Season Averages** από `players_<SEASON>_<MODE>.csv`. "
    "Για τα **Game Logs**, αν υπάρχει το `player_gamelogs_<SEASON>_<MODE>.csv`, "
    "εμφανίζονται αναλυτικά ανά παίκτη με βάση την επιλογή σου."
)
