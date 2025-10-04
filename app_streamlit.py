import os
import sys
import subprocess
import pandas as pd
import streamlit as st

# --------- Βασικές ρυθμίσεις σελίδας ----------
st.set_page_config(page_title="EuroLeague Fantasy Stats", layout="wide")
st.title("EuroLeague Fantasy – Player Stats")
st.caption("Live στο cloud, με απλό password & αυτόματη λήψη δεδομένων")

# --------- Απλό password gate ----------
PASS = st.secrets.get("APP_PASSWORD", None)   # θα το βάλεις στα Streamlit Secrets
if PASS:
    with st.sidebar:
        pwd = st.text_input("Password", type="password")
    if pwd != PASS:
        st.warning("🔒 Βάλε σωστό password για να προχωρήσεις.")
        st.stop()

# --------- Διαδρομές / ρυθμίσεις ----------
OUT_DIR = "out"
DEFAULT_SEASONS = os.getenv("SEASONS", "2025")         # μπορείς να αλλάξεις από Secrets αργότερα
DEFAULT_MODE = os.getenv("STAT_MODE", "perGame")       # perGame | perMinute | accumulated

# --------- Αν δεν υπάρχουν CSV, κατέβασέ τα αυτόματα ----------
def ensure_data():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)
    csv_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".csv")]
    if csv_files:
        return

    with st.spinner("⏳ Δεν βρέθηκαν δεδομένα — γίνεται λήψη από EuroLeague API..."):
        try:
            subprocess.run(
                [
                    sys.executable,
                    "fetch_euroleague_stats.py",
                    "--seasons", *DEFAULT_SEASONS.split(),
                    "--mode", DEFAULT_MODE,
                    "--out", OUT_DIR,
                ],
                check=True,
            )
            st.success("✅ Ολοκληρώθηκε η λήψη δεδομένων.")
        except Exception as e:
            st.error(f"❌ Αποτυχία λήψης δεδομένων: {e}")
            st.stop()

ensure_data()

# --------- Φόρτωση και φίλτρα ----------
csv_files = sorted([f for f in os.listdir(OUT_DIR) if f.endswith(".csv")])
if not csv_files:
    st.error("Δεν υπάρχουν διαθέσιμα datasets στο 'out/'. Δοκίμασε να ξαναφορτώσεις τη σελίδα.")
    st.stop()

fname = st.selectbox("Διάλεξε dataset", csv_files)
df = pd.read_csv(os.path.join(OUT_DIR, fname))

left, right = st.columns([2, 1])
with right:
    team_col = "Team" if "Team" in df.columns else ("team_name" if "team_name" in df.columns else None)
    team_list = ["(Όλες)"]
    if team_col:
        team_list += sorted([t for t in df[team_col].dropna().unique()])
    team = st.selectbox("Ομάδα", team_list)

    gp_col = "GP" if "GP" in df.columns else ("gp" if "gp" in df.columns else None)
    min_gp = st.number_input("Min GP", min_value=0, value=0)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    sort_by = st.selectbox("Ταξινόμηση", numeric_cols or ["PTS"])

# Εφαρμογή φίλτρων
fdf = df.copy()
if team_col and team != "(Όλες)":
    fdf = fdf[fdf[team_col] == team]
if gp_col:
    fdf = fdf[fdf[gp_col] >= min_gp]

fdf = fdf.sort_values(by=sort_by, ascending=False)

st.dataframe(fdf, use_container_width=True)
st.download_button("Λήψη (CSV)", data=fdf.to_csv(index=False), file_name="filtered.csv", mime="text/csv")
