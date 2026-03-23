import pandas as pd
import re

YEAR = 2026
INPUT_CSV = f"data/raw/tarifdouanier_{YEAR}_raw.csv"
OUTPUT_CSV = f"data/processed/tarifdouanier_{YEAR}_clean.csv"


def normalize_label(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_hs(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\D", "", s)
    return s


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    df["hs_code"] = df["hs_code"].map(normalize_hs)
    df["label"] = df["label"].map(normalize_label)

    # on garde surtout les codes utiles pour ton moteur
    df = df[df["hs_code"].str.len().isin([6, 8, 10])]
    df = df[df["label"] != ""]

    # suppression doublons
    df = df.drop_duplicates(subset=["hs_code", "label"]).copy()

    # colonnes standardisées pour la fusion
    df["SOURCE"] = "TARIF_DOUANIER"
    df["DESCRIPTION"] = df["label"]
    df["HS_CODE"] = df["hs_code"]

    out = df[["DESCRIPTION", "HS_CODE", "SOURCE", "year", "page_url"]].copy()
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Clean rows: {len(out)}")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()