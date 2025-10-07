#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, time, re, unicodedata
import pandas as pd
import requests

# ---- helpers ----
def slugify(name: str) -> str:
    if not name: return ""
    name = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in name if not unicodedata.combining(c)]).lower()
    name = re.sub(r"[^a-z0-9]+","-",name).strip("-")
    return re.sub(r"-+","-",name)

def first_last(name: str) -> str:
    if not name: return ""
    parts=[p for p in re.split(r"\s+", name.strip()) if p]
    if not parts: return ""
    return parts[0] if len(parts)==1 else f"{parts[0]} {parts[-1]}"

def choose_col(df: pd.DataFrame, options):
    for c in options:
        if c in df.columns: return c
    # χαλαρό matching (case-insensitive, χωρίς κενά)
    norm = {re.sub(r"\s+","",str(c)).lower(): c for c in df.columns}
    for opt in options:
        k = re.sub(r"\s+","",opt).lower()
        if k in norm: return norm[k]
    return None

def compose_name(df: pd.DataFrame):
    # Αν δεν υπάρχει μια "ενιαία" στήλη ονόματος, φτιάξε από first/last
    first_opts = ["FirstName","First Name","first_name","FIRSTNAME","FIRST NAME","Name","First"]
    last_opts  = ["LastName","Last Name","last_name","LASTNAME","LAST NAME","Surname","Last"]
    fc = choose_col(df, first_opts)
    lc = choose_col(df, last_opts)
    if fc and lc:
        return (df[fc].fillna("").astype(str) + " " + df[lc].fillna("").astype(str)).str.strip()
    return None

def url_candidates(base_lang: str, comp_path: str, slug: str, code: str):
    yield f"https://www.euroleaguebasketball.net/{base_lang}/{comp_path}/players/{slug}/{code}/"
    yield f"https://www.euroleaguebasketball.net/{base_lang}/{comp_path}/players/{slug}/{code}"
    last = slug.split("-")[-1] if slug else ""
    if last:
        yield f"https://www.euroleaguebasketball.net/{base_lang}/{comp_path}/players/{last}/{code}/"

def probe_url(session: requests.Session, urls):
    for u in urls:
        try:
            r = session.head(u, allow_redirects=True, timeout=12)
            if r.status_code in (403,404,405):
                r = session.get(u, allow_redirects=True, timeout=12)
            if 200 <= r.status_code < 400:
                return r.url
        except requests.RequestException:
            pass
    return ""

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="E2025")
    ap.add_argument("--competition", default="E", help="E=EuroLeague, U=EuroCup")
    ap.add_argument("--infile", default=None)
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--lang", default="el", choices=["el","en"])
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    comp = args.competition.upper()
    comp_path = "euroleague" if comp.startswith("E") else "eurocup"

    # εντοπισμός infile
    infile = args.infile or f"out/players_{args.season}_perGame.csv"
    if not os.path.exists(infile):
        alt = f"out/players_{str(args.season).lstrip('EU')}_perGame.csv"
        if os.path.exists(alt):
            infile = alt
    if not os.path.exists(infile):
        raise SystemExit(f"Δεν βρέθηκε input CSV: {infile}")

    df = pd.read_csv(infile)

    # Πιο πλούσιες λίστες ονομάτων στηλών
    code_opts = ["player_code","playerId","PlayerId","PLAYERID","id","ID","Code","code","Player Code","PlayerCode"]
    
    name_opts = [
    "Player","player","name","Name",
    "FullName","FULLNAME","full_name","Full Name",
    "player_name"   # <-- πρόσθεσέ το εδώ
    ]


    code_col = choose_col(df, code_opts)
    name_col = choose_col(df, name_opts)

    # αν δεν υπάρχει ενιαία στήλη name, προσπάθησε να τη φτιάξεις
    if not name_col:
        composed = compose_name(df)
        if composed is not None:
            name_col = "_COMPOSED_NAME_"
            df[name_col] = composed

    if not code_col or not name_col:
        raise SystemExit(
            "Δεν βρέθηκαν απαιτούμενες στήλες. Βρέθηκαν columns: "
            + ", ".join(map(str, df.columns))
            + "\nΨάχνω code σε: " + str(code_opts)
            + "\nΨάχνω name σε: " + str(name_opts) + " ή First/Last."
        )

    session = requests.Session()
    session.headers.update({"User-Agent":"eurol-url-builder/1.1","Accept":"text/html"})

    out_rows = []
    seen = set()
    for _, r in df[[code_col, name_col]].dropna().iterrows():
        code = str(r[code_col]).strip()
        name = str(r[name_col]).strip()
        key = (code, name)
        if key in seen:
            continue
        seen.add(key)

        slugs = []
        s_full = slugify(name);  s_fl = slugify(first_last(name))
        for s in (s_full, s_fl):
            if s and s not in slugs: slugs.append(s)

        found = ""
        for s in slugs or [""]:
            found = probe_url(session, list(url_candidates(args.lang, comp_path, s, code)))
            if found: break

        out_rows.append({"player_code": code, "Player": name, "player_url": found})
        time.sleep(args.sleep)

    out_df = pd.DataFrame(out_rows)
    outfile = args.outfile or f"out/player_urls_{args.season}.csv"
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    out_df.to_csv(outfile, index=False, encoding="utf-8-sig")
    ok = int(out_df["player_url"].astype(bool).sum())
    print(f"[OK] wrote {outfile} rows={len(out_df)} resolved={ok}")

if __name__ == "__main__":
    main()
