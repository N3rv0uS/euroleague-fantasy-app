#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, time, pandas as pd, requests
from bn_probe import parse_bn_html_gamelogs, unify_columns

UA = {"User-Agent": "Mozilla/5.0"}

def fetch_page(url: str, timeout=30) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--competition", default="E")
    ap.add_argument("--mode", default="perGame")
    ap.add_argument("--map_csv", default="bn_map.csv", help="CSV: player_code,bn_path")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay_ms", type=int, default=300)
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.map_csv):
        raise SystemExit(f"Λείπει {args.map_csv}. Φτιάξ' το με bn_make_mapping.py και συμπλήρωσε bn_path.")

    m = pd.read_csv(args.map_csv)
    if not {"player_code","bn_path"} <= set(m.columns):
        raise SystemExit("Το mapping πρέπει να έχει στήλες: player_code, bn_path")

    subset = m[m["bn_path"].astype(str).str.strip() != ""]
    if args.limit > 0:
        subset = subset.head(args.limit)

    frames = []
    for i, row in enumerate(subset.itertuples(index=False), start=1):
        code = str(row.player_code)
        bn_path = str(row.bn_path).strip()
        page_url = f"https://basketnews.com/players/{bn_path}/statistics/leagues/25-euroleague/{args.season}.html"
        try:
            html = fetch_page(page_url)
            df = parse_bn_html_gamelogs(html)
            if df.empty:
                print(f"[{i}] {code} -> 0 rows")
            else:
                df["player_code"] = code
                df["season"] = args.season
                df["competition"] = args.competition
                df["mode"] = args.mode
                frames.append(df)
                print(f"[{i}] {code} -> +{len(df)} rows")
        except Exception as e:
            print(f"[{i}] {code} ERROR: {e}")
        time.sleep(args.delay_ms / 1000.0)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = unify_columns(out)  # τελικό safety
        out_csv = os.path.join(args.out_dir, f"player_gamelogs_{args.season}_{args.mode}.csv")
        out.to_csv(out_csv, index=False)
        print(f"[OK] saved {out_csv} rows={len(out)}")
    else:
        print("[WARN] Δεν μαζεύτηκαν gamelogs.")

if __name__ == "__main__":
    main()
