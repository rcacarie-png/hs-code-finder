import pandas as pd

CLUBMED_FILE = "data/processed/clubmed_hs_reference.csv"
TARIF_FILE = "data/processed/tarifdouanier_2026_clean.csv"
OUTPUT_FILE = "data/merged/hs_reference_merged.csv"


def main():
    clubmed = pd.read_csv(CLUBMED_FILE, dtype=str).fillna("")
    tarif = pd.read_csv(TARIF_FILE, dtype=str).fillna("")

    # Harmonisation minimale
    if "DESCRIPTION" not in clubmed.columns:
        raise ValueError("clubmed_hs_reference.csv doit contenir une colonne DESCRIPTION")
    if "HS_CODE" not in clubmed.columns:
        raise ValueError("clubmed_hs_reference.csv doit contenir une colonne HS_CODE")

    merged = pd.concat([clubmed, tarif], ignore_index=True)

    # Club Med prioritaire en cas de doublon exact sur description
    merged["priority"] = merged["SOURCE"].map({
        "CLUBMED": 1,
        "TARIF_DOUANIER": 2
    }).fillna(99)

    merged = merged.sort_values(["DESCRIPTION", "priority"])
    merged = merged.drop_duplicates(subset=["DESCRIPTION", "HS_CODE"], keep="first")

    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Merged rows: {len(merged)}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()