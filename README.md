# EuroLeague Fantasy – Starter

Αυτό το πακέτο σου δίνει **έτοιμο κώδικα** για να κατεβάζεις όλα τα player stats της EuroLeague (όπως στη σελίδα "Expanded Stats") και να τα δουλεύεις σε **CSV / Excel / SQLite**, καθώς και ένα μικρό **Streamlit** app για γρήγορα φίλτρα.

## Τι περιλαμβάνει
- `fetch_euroleague_stats.py`: Script που κατεβάζει συγκεντρωτικά player stats ανά σεζόν από **το επίσημο EuroLeague API** (μέσω του community πακέτου `euroleague-api`) και τα αποθηκεύει σε CSV/Excel/SQLite.
- `app_streamlit.py`: Μικρό web app (τοπικά) για να φιλτράρεις/επεξεργάζεσαι τα στατιστικά.
- `requirements.txt`: Βασικές εξαρτήσεις.
- `config.json`: Ρυθμίσεις (σεζόν, competition code κ.λπ.).

> Σημείωση: Το API φορτώνει τα δεδομένα που βλέπεις και στο site. Αν το package `euroleague-api` αλλάξει endpoints ή σπάσει, το script έχει και **fallback** με `requests` για δημοφιλή endpoints.

## Γρήγορη εκτέλεση
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Κατέβασμα stats για σεζόν που θες (δες config.json)
python fetch_euroleague_stats.py --seasons 2025 2024 --mode perGame --out out/

# Άνοιγμα του μίνι app
streamlit run app_streamlit.py
```

### Έξοδοι
- `out/players_2025_perGame.csv` (και .xlsx)
- `out/euroleague.db` (SQLite) με πίνακα `player_stats`

## Παραμετροποίηση
- Άλλαξε στο `config.json`:
  - `competition_code`: `"E"` για EuroLeague, `"U"` για EuroCup
  - `seasons`: λίστα από ακέραια έτη (π.χ. `2025`)
  - `statistic_mode`: `perGame` | `perMinute` | `accumulated`

## Νομικό
Τα δεδομένα ανήκουν στον κάτοχο τους (EuroLeague). Χρησιμοποίησέ τα σύμφωνα με τους όρους χρήσης.
