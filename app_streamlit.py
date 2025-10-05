# app_streamlit.py
import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
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


def safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return (num / den).fillna(0.0)


def clamp01(s: pd.Series) -> pd.Series:
    return s.clip(lower=0, upper=1)


def minmax_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s_clipped = s.clip(lower=s.quantile(0.02), upper=s.quantile(0.98))  # αντοχή σε outliers
    rng = s_clipped.max() - s_clipped.min()
    if rng == 0:
        return pd.Series(50.0, index=s.index)
    return 100.0 * (s_clipped - s_clipped.min()) / rng


# ---------- NORMALIZATION ----------
def normalize_players_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    rename_map: Dict[str, str] = {}

    # IDs / meta
    for a in ["player_code", "code", "playerCode", "id"]:
        if a in df.columns: rename_map[a] = "player_code"
    for a in ["player_name", "name", "playerName"]:
        if a in df.columns: rename_map[a] = "Player"
    for a in ["team_code", "player_team_code", "teamCode", "team", "Team code"]:
        if a in df.columns: rename_map[a] = "Team"
    for a in ["team_name", "player_team_name", "teamName"]:
        if a in df.columns and "Team" not in rename_map.values(): rename_map[a] = "Team"

    # Position (θα χρησιμοποιηθεί στα recommendations)
    for a in ["player_position", "position", "positionShort", "pos"]:
        if a in df.columns: rename_map[a] = "PosRaw"

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

    # PTS
    for a in ["pointsScored", "PTS", "Points"]:
        if a in df.columns: rename_map[a] = "PTS"

    # 2P
    if "twoPointersMade" in df.columns: rename_map["twoPointersMade"] = "2PM"
    if "twoPointersAttempted" in df.columns: rename_map["twoPointersAttempted"] = "2PA"
    if "twoPointersPercentage" in df.columns: rename_map["twoPointersPercentage"] = "2P%"
    if "2PM" in df.columns: rename_map["2PM"] = "2PM"
    if "2PA" in df.columns: rename_map["2PA"] = "2PA"
    if "2P%" in df.columns: rename_map["2P%"] = "2P%"

    # 3P
    if "threePointersMade" in df.columns: rename_map["threePointersMade"] = "3PM"
    if "threePointersAttempted" in df.columns: rename_map["threePointersAttempted"] = "3PA"
    if "threePointersPercentage" in df.columns: rename_map["threePointersPercentage"] = "3P%"
    if "3PM" in df.columns: rename_map["3PM"] = "3PM"
    if "3PA" in df.columns: rename_map["3PA"] = "3PA"
    if "3P%" in df.columns: rename_map["3P%"] = "3P%"

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

    # Playmaking / defense / fouls
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

    # Season / competition
    for a in ["season", "Season"]:
        if a in df.columns: rename_map[a] = "season"
    for a in ["competition", "Competition"]:
        if a in df.columns: rename_map[a] = "competition"

    df = df.rename(columns=rename_map)

    # Αντιμετώπιση ελαχίστων
    for c in ["Player", "player_code", "Team"]:
        if c not in df.columns: df[c] = None

    if "TR" not in df.columns and all(c in df.columns for c in ["OR", "DR"]):
        df["TR"] = df["OR"].fillna(0) + df["DR"].fillna(0)

    target_cols = [
        "Player", "Team",
        "GP", "GS", "Min", "PTS",
        "2PM", "2PA", "2P%",
        "3PM", "3PA", "3P%",
        "FTM", "FTA", "FT%",
        "OR", "DR", "TR",
        "AST", "STL", "TO", "BLK", "BLKA",
        "FC", "FD", "PIR",
        "BCI", "Stability", "Form3",  # advanced
    ]

    keep = [c for c in target_cols if c in df.columns and c not in ["BCI", "Stability", "Form3"]]
    if "PIR" in df.columns:
        df = df.sort_values("PIR", ascending=False, na_position="last")

    return df, keep, target_cols


def normalize_gamelogs_df(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for a in ["player_code", "code", "playerCode", "id"]:
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
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

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

    # σκοπεύουμε να υπολογίσουμε και Stocks
    if "Stocks" not in df.columns and all(c in df.columns for c in ["STL", "BLK"]):
        df["Stocks"] = df["STL"].fillna(0) + df["BLK"].fillna(0)

    return df


# ---------- ADVANCED / FEATURES ----------
def compute_advanced(players_df: pd.DataFrame, gl_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = players_df.copy()

    # -- BCI (από προηγούμενη λογική)
    PTS_pm = safe_div(out.get("PTS", 0), out.get("Min", 0))
    TR_pm  = safe_div(out.get("TR", 0),  out.get("Min", 0))
    AST_pm = safe_div(out.get("AST", 0), out.get("Min", 0))
    FD_pm  = safe_div(out.get("FD", 0),  out.get("Min", 0))
    PIR_pm = safe_div(out.get("PIR", 0), out.get("Min", 0))

    raw_bci = 0.35*PTS_pm + 0.25*TR_pm + 0.25*AST_pm + 0.10*FD_pm + 0.05*PIR_pm
    min_bonus = (out.get("Min", 0).fillna(0) / 30.0).clip(0.6, 1.2)
    raw_bci = raw_bci * min_bonus
    out["BCI"] = (100 * safe_div(raw_bci, raw_bci.max())).round(1)

    # -- Shooting totals (proxy) για efficiency
    FGA = out.get("2PA", 0).fillna(0) + out.get("3PA", 0).fillna(0)
    FGM = out.get("2PM", 0).fillna(0) + out.get("3PM", 0).fillna(0)
    FTA = out.get("FTA", 0).fillna(0)
    FTM = out.get("FTM", 0).fillna(0)

    # -- eFG% & TS%
    eFG = safe_div(out.get("2PM", 0)*2 + out.get("3PM", 0)*3, FGA*2).replace([np.inf, -np.inf], 0)
    TS  = safe_div(out.get("PTS", 0), (2*(FGA + 0.44*FTA))).replace([np.inf, -np.inf], 0)

    out["eFG%"] = (eFG*100).round(1)
    out["TS%"]  = (TS*100).round(1)

    # -- FT Rate
    out["FT_rate"] = safe_div(FTA, FGA).round(3)

    # -- Per-minute core
    out["PTS_pm"] = PTS_pm.round(3)
    out["TR_pm"]  = TR_pm.round(3)
    out["AST_pm"] = AST_pm.round(3)
    out["FD_pm"]  = FD_pm.round(3)
    out["TO_pm"]  = safe_div(out.get("TO", 0), out.get("Min", 0)).round(3)
    out["OR_pm"]  = safe_div(out.get("OR", 0), out.get("Min", 0)).round(3)
    out["DR_pm"]  = safe_div(out.get("DR", 0), out.get("Min", 0)).round(3)
    out["Stocks_pm"] = safe_div(out.get("STL", 0).fillna(0) + out.get("BLK", 0).fillna(0), out.get("Min", 0)).round(3)

    # -- Usage proxy (per min)
    out["USG_pm"] = safe_div(FGA + 0.44*FTA + out.get("TO", 0).fillna(0), out.get("Min", 0)).round(3)

    # -- Stability & Form3 από gamelogs
    out["Stability"] = np.nan
    out["Form3"] = np.nan
    if gl_df is not None and not gl_df.empty:
        key = "player_code" if "player_code" in gl_df.columns and "player_code" in out.columns else "Player"
        if key in gl_df.columns and key in out.columns:
            for pid, g in gl_df.groupby(key):
                g = g.sort_values("game_date")
                pir = pd.to_numeric(g.get("PIR"), errors="coerce")
                if pir is None or pir.dropna().empty:
                    continue
                last6 = pir.dropna().tail(6)
                if len(last6) >= 3 and last6.mean() != 0:
                    cv = last6.std(ddof=0) / abs(last6.mean())
                    stab = (1.0 / (1.0 + cv)) * 100.0
                    out.loc[out[key] == pid, "Stability"] = stab
                last3 = pir.dropna().tail(3)
                if len(last3) > 0:
                    out.loc[out[key] == pid, "Form3"] = last3.mean()
    out["Stability"] = out["Stability"].round(1)
    out["Form3"] = out["Form3"].round(1)

    # -- Κανονικοποιήσεις για scoring (0–100)
    n_MIN      = minmax_0_100(out.get("Min", 0))
    n_TS       = minmax_0_100(out.get("TS%", 0))
    n_eFG      = minmax_0_100(out.get("eFG%", 0))
    n_USGpm    = minmax_0_100(out.get("USG_pm", 0))
    n_ASTpm    = minmax_0_100(out.get("AST_pm", 0))
    n_TRpm     = minmax_0_100(out.get("TR_pm", 0))
    n_PTSpm    = minmax_0_100(out.get("PTS_pm", 0))
    n_FDpm     = minmax_0_100(out.get("FD_pm", 0))
    n_Stocks   = minmax_0_100(out.get("Stocks_pm", 0))
    n_TOpm_inv = 100 - minmax_0_100(out.get("TO_pm", 0))  # χαμηλό TO_pm = καλύτερο
    n_2Ppct    = minmax_0_100(out.get("2P%", 0))
    n_Form3    = minmax_0_100(out.get("Form3", 0).fillna(0))
    n_Stab     = minmax_0_100(out.get("Stability", 0).fillna(0))

    # ---------- ΘΕΣΗ ΠΑΙΚΤΗ ----------
    pos = out.get("PosRaw")
    if pos is None:
        out["Pos"] = "Unknown"
    else:
        pr = pos.astype(str).str.upper().fillna("")
        cond_C = pr.str.contains("C")
        cond_F = pr.str.contains("F")
        cond_G = pr.str.contains("G")
        out["Pos"] = np.where(cond_C, "C", np.where(cond_F, "F", np.where(cond_G, "G", "Unknown")))

    # ---------- POSITION SCORES ----------
    # Guards: MIN, USG, AST, TS, FD, Stocks, -TO, +Form/Stab
    out["GuardScore"] = (
        0.25*n_MIN + 0.20*n_USGpm + 0.20*n_ASTpm + 0.15*n_TS + 0.10*n_FDpm +
        0.10*n_Stocks + 0.10*n_TOpm_inv + 0.05*n_Form3 + 0.05*n_Stab
    )

    # Forwards: MIN, PTS_pm, TR_pm, TS, FD, Stocks, -TO, +Form/Stab
    out["ForwardScore"] = (
        0.25*n_MIN + 0.18*n_PTSpm + 0.18*n_TRpm + 0.12*n_TS + 0.10*n_FDpm +
        0.10*n_Stocks + 0.07*n_TOpm_inv + 0.05*n_Form3 + 0.05*n_Stab
    )

    # Centers: MIN, TR_pm, 2P%, BLK/Stocks, FD, -TO, +Form/Stab
    # (Δίνουμε έμφαση σε 2P% αντί για TS, και λίγο παραπάνω σε Stocks)
    out["CenterScore"] = (
        0.25*n_MIN + 0.25*n_TRpm + 0.15*n_2Ppct + 0.15*n_Stocks + 0.10*n_FDpm +
        0.10*n_TOpm_inv + 0.05*n_Form3 + 0.05*n_Stab
    )

    # Στρογγυλοποιήσεις
    for c in ["GuardScore", "ForwardScore", "CenterScore"]:
        out[c] = out[c].round(1)

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

players_df_norm, base_cols, target_cols = normalize_players_df(players_df_raw)

gamelogs_df = None
if gamelogs_df_raw is not None and not gamelogs_df_raw.empty:
    gamelogs_df = normalize_gamelogs_df(gamelogs_df_raw.copy())

players_df = compute_advanced(players_df_norm, gamelogs_df)

# Εμφάνισε πάντα στήλες Stability/Form3
for _col in ["Stability", "Form3"]:
    if _col not in players_df.columns:
        players_df[_col] = np.nan

# Τελικές στήλες (season table)
final_cols = []
for c in target_cols:
    if c in players_df.columns:
        final_cols.append(c)

# Προσθέτουμε και τα νέα advanced στο season table (μπορείς να αφαιρέσεις όποια δεν θες)
season_extra = [
    "TS%", "eFG%", "FT_rate",
    "USG_pm", "PTS_pm", "TR_pm", "AST_pm", "FD_pm", "TO_pm", "Stocks_pm",
]
for c in season_extra:
    if c in players_df.columns and c not in final_cols:
        final_cols.append(c)

teams_list = ["(Όλες)"] + sorted(players_df.get("Team", pd.Series(dtype=str)).dropna().astype(str).unique(), key=lambda x: x.lower())
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    q = st.text_input("🔎 Live search (όνομα/κωδικός/ομάδα)", "")
with f2:
    team_sel = st.selectbox("Ομάδα", teams_list, index=0)
with f3:
    min_gp = st.number_input("Min GP", min_value=0, max_value=50, value=0, step=1)

filtered_players = filter_players(players_df, q, team_sel, min_gp)

st.subheader("Season Averages (με ζητούμενες στήλες + Advanced)")
st.dataframe(
    filtered_players[final_cols].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

# ---------- Επιλογή παίκτη & Game-by-Game ----------
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
                p_name = sel_row["Player"]
                player_gl = gamelogs_df[gamelogs_df["Player"].astype(str).str.lower() == str(p_name).lower()]

        if selected_label and not player_gl.empty:
            csum1, csum2, csum3 = st.columns(3)
            with csum1: st.metric("Games", len(player_gl))
            with csum2: st.metric("PTS (avg)", round(player_gl.get("PTS", pd.Series([0])).mean(), 2))
            with csum3: st.metric("PIR (avg)", round(player_gl.get("PIR", pd.Series([0])).mean(), 2))

            # Charts
            if "game_date" in player_gl.columns:
                if "PTS" in player_gl.columns:
                    sub = player_gl[["game_date", "PTS"]].dropna().sort_values("game_date").set_index("game_date")
                    st.line_chart(sub, height=180, use_container_width=True)
                if "PIR" in player_gl.columns:
                    sub = player_gl[["game_date", "PIR"]].dropna().sort_values("game_date").set_index("game_date")
                    st.line_chart(sub, height=180, use_container_width=True)

            gl_cols_pref = ["game_date", "Team", "opponent", "MIN", "PTS", "TR", "AST", "STL", "TO", "BLK", "FC", "FD", "PIR"]
            gl_cols = [c for c in gl_cols_pref if c in player_gl.columns]
            st.dataframe(player_gl[gl_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            if selected_label:
                st.info("Δεν βρέθηκαν gamelogs για τον συγκεκριμένο παίκτη (ή το αρχείο είναι κενό).")

st.divider()

# ---------- RECOMMENDATIONS ----------
st.header("🔮 Position-based Recommendations (PIR potential)")

def rec_table(df: pd.DataFrame, pos_letter: str, topn: int, score_col: str, title: str):
    sub = df[df.get("Pos") == pos_letter].copy()
    if sub.empty:
        st.warning(f"Δεν βρέθηκαν παίκτες με θέση {pos_letter}.")
        return
    cols_show = [
        "Player", "Team", "Pos", score_col, "Min", "Form3", "Stability",
        "PTS_pm", "TR_pm", "AST_pm", "TS%", "FD_pm", "TO_pm", "Stocks_pm", "BCI", "PIR"
    ]
    cols = [c for c in cols_show if c in sub.columns]
    sub = sub.sort_values(score_col, ascending=False).head(topn)
    st.subheader(title)
    st.dataframe(sub[cols].reset_index(drop=True), use_container_width=True, hide_index=True)

# 10 Centers, 15 Forwards, 20 Guards
rec_table(players_df, "C", 10, "CenterScore",  "Top Centers (10)")
rec_table(players_df, "F", 15, "ForwardScore", "Top Forwards (15)")
rec_table(players_df, "G", 20, "GuardScore",   "Top Guards (20)")

st.caption(
    "Οι βαθμολογίες ανά θέση ζυγίζουν διαφορετικά τα features (λεπτά, usage, αποδοτικότητα, "
    "ριμπάουντ/δημιουργία/στοκς, λάθη, καθώς και πρόσφατη φόρμα/σταθερότητα). "
    "Αν λείπει τελείως η θέση από το dataset, χαρτογραφείται ως 'Unknown' και ο παίκτης δεν "
    "εμφανίζεται στα position tables."
)
