import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# Optional (web lookup)
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus


# ----------------------------
# Helpers: normalize strings
# ----------------------------
def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    # remove accents
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    # keep alnum + spaces
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find a column by fuzzy header matching (case-insensitive)."""
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fallback: contains
    for c in df.columns:
        cl = str(c).lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def find_hs_col(df: pd.DataFrame) -> Optional[str]:
    # Typical headers: "HS CODE", "HS COD", "HS Code"
    for c in df.columns:
        cl = str(c).lower().replace(" ", "")
        if "hs" in cl and "cod" in cl:
            return c
    return None


# ----------------------------
# Web lookup (best-effort)
# ----------------------------
HS_PATTERN = re.compile(r"\b\d{4}(\.\d{2}){1,3}\b|\b\d{6,10}\b")

def lookup_hs_web_best_effort(query: str, timeout: int = 12) -> Optional[str]:
    """
    Best-effort scrape.
    IMPORTANT: you may need to adjust the search URL depending on tarifdouanier.eu routing.
    """
    q = normalize_text(query)
    if not q:
        return None

    # Try a plausible search endpoint (may need adjustment).
    # If it fails, we still won't crash the app.
    search_url = f"https://www.tarifdouanier.eu/recherche?query={quote_plus(q)}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(search_url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        # pick first plausible HS code-ish token
        m = HS_PATTERN.search(text)
        if not m:
            return None

        hs = m.group(0)
        # Normalize: keep digits only if needed
        hs_digits = re.sub(r"\D", "", hs)
        if len(hs_digits) >= 6:
            return hs_digits
        return hs
    except Exception:
        return None


# ----------------------------
# Matching engine
# ----------------------------
@dataclass
class MatchResult:
    hs_code: Optional[str]
    match_type: str          # EXACT / FUZZY / WEB / NOT_FOUND / FOUND_NO_HS_IN_BDD
    source: str              # BDD / WEB / NONE
    detail: str


@st.cache_data(show_spinner=False)
def load_bdd(bdd_file_bytes: bytes) -> pd.DataFrame:
    # Read first sheet by default
    df = pd.read_excel(io.BytesIO(bdd_file_bytes), sheet_name=0, engine="openpyxl")
    # Expected columns in your BDD: FOURNISSEUR, CODE ARTICLE, Description, HS CODE, Origine fabrication
    # We'll normalize names a bit
    df.columns = [str(c).strip() for c in df.columns]

    # Create normalized helper columns
    code_col = find_col(df, ["CODE ARTICLE", "CODE ARTICLE ", "CODE_ARTICLE", "ARTICLE"])
    desc_col = find_col(df, ["Description", "DESIGNATION", "LIBELLE"])
    hs_col = find_col(df, ["HS CODE", "HS", "HS_CODE"])

    if code_col is None:
        df["_CODE_N"] = ""
    else:
        df["_CODE_N"] = df[code_col].astype(str).map(normalize_text)

    if desc_col is None:
        df["_DESC_N"] = ""
    else:
        df["_DESC_N"] = df[desc_col].astype(str).map(normalize_text)

    if hs_col is None:
        df["_HS"] = None
    else:
        df["_HS"] = df[hs_col]

    # Keep only rows that have at least something
    df = df.fillna("")
    return df


def build_bdd_indexes(bdd: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    # code -> hs (prefer first non-empty)
    code_to_hs: Dict[str, str] = {}
    desc_to_hs: Dict[str, str] = {}
    desc_list: List[str] = []

    for _, row in bdd.iterrows():
        code = row.get("_CODE_N", "")
        desc = row.get("_DESC_N", "")
        hs = row.get("_HS", "")
        hs_str = str(hs).strip()
        if hs_str == "" or hs_str.lower() == "nan":
            hs_str = ""

        if code and code not in code_to_hs and hs_str:
            code_to_hs[code] = hs_str

        if desc:
            desc_list.append(desc)
            if desc not in desc_to_hs and hs_str:
                desc_to_hs[desc] = hs_str

    return code_to_hs, desc_to_hs, desc_list


def match_row(
    article_code: str,
    libelle: str,
    code_to_hs: Dict[str, str],
    desc_to_hs: Dict[str, str],
    desc_list: List[str],
    fuzzy_threshold: int,
    enable_web: bool
) -> MatchResult:
    code_n = normalize_text(article_code)
    lib_n = normalize_text(libelle)

    # 1) Exact by code
    if code_n:
        hs = code_to_hs.get(code_n)
        if hs:
            return MatchResult(hs, "EXACT", "BDD", "CODE ARTICLE exact")
        # code found in BDD but HS empty is hard to know without extra index;
        # we still try description routes next.

    # 2) Exact by normalized description
    if lib_n and lib_n in desc_to_hs:
        return MatchResult(desc_to_hs[lib_n], "EXACT", "BDD", "Description exact")

    # 3) Fuzzy
    if lib_n and desc_list:
        best = process.extractOne(
            lib_n,
            desc_list,
            scorer=fuzz.token_sort_ratio
        )
        if best:
            match_desc, score, _ = best
            if score >= fuzzy_threshold:
                hs = desc_to_hs.get(match_desc)
                if hs:
                    return MatchResult(
                        hs,
                        "FUZZY",
                        "BDD",
                        f'fuzzy score={score}; matched="{match_desc[:80]}"'
                    )
                else:
                    # matched row exists but HS missing in BDD
                    if enable_web:
                        hs_web = lookup_hs_web_best_effort(libelle)
                        if hs_web:
                            return MatchResult(hs_web, "WEB", "WEB", f"web from fuzzy-matched item; score={score}")
                    return MatchResult(None, "FOUND_NO_HS_IN_BDD", "BDD", f"fuzzy matched but HS missing; score={score}")

    # 4) Web fallback
    if enable_web and libelle:
        hs_web = lookup_hs_web_best_effort(libelle)
        if hs_web:
            return MatchResult(hs_web, "WEB", "WEB", "web lookup")

    return MatchResult(None, "NOT_FOUND", "NONE", "no match")


def process_workbook(
    input_bytes: bytes,
    bdd_bytes: bytes,
    fuzzy_threshold: int,
    enable_web: bool
) -> bytes:
    bdd = load_bdd(bdd_bytes)
    code_to_hs, desc_to_hs, desc_list = build_bdd_indexes(bdd)

    xls = pd.ExcelFile(io.BytesIO(input_bytes), engine="openpyxl")
    out_buffer = io.BytesIO()

    with pd.ExcelWriter(out_buffer, engine="openpyxl") as writer:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            hs_col = find_hs_col(df)
            if hs_col is None:
                # write untouched
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            # identify useful columns
            article_col = find_col(df, ["Article", "CODE ARTICLE", "Code article"])
            lib_col = find_col(df, ["Libellé à imprimer", "Libelle a imprimer", "Description", "Libellé", "Libelle"])

            if lib_col is None:
                # If no text column, we still try code-only
                lib_col = None

            # Ensure metadata columns exist right after HS column
            # We'll append then reorder to place them immediately after hs_col.
            meta_cols = ["HS_MATCH_TYPE", "HS_SOURCE", "HS_MATCH_DETAIL"]
            for mc in meta_cols:
                if mc not in df.columns:
                    df[mc] = ""

            # Fill
            for i in range(len(df)):
                current_hs = str(df.at[i, hs_col]).strip() if i in df.index else ""
                if current_hs and current_hs.lower() != "nan":
                    # Already filled: keep it, but mark as already present
                    df.at[i, "HS_MATCH_TYPE"] = "ALREADY_PRESENT"
                    df.at[i, "HS_SOURCE"] = "INPUT"
                    df.at[i, "HS_MATCH_DETAIL"] = "HS already filled in input file"
                    continue

                article_val = str(df.at[i, article_col]).strip() if article_col else ""
                lib_val = str(df.at[i, lib_col]).strip() if lib_col else ""

                res = match_row(
                    article_code=article_val,
                    libelle=lib_val,
                    code_to_hs=code_to_hs,
                    desc_to_hs=desc_to_hs,
                    desc_list=desc_list,
                    fuzzy_threshold=fuzzy_threshold,
                    enable_web=enable_web
                )

                hscode = res.hs_code
                if hscode:
                    df.at[i, hs_col] = hscode


                df.at[i, "HS_MATCH_TYPE"] = res.match_type
                df.at[i, "HS_SOURCE"] = res.source
                df.at[i, "HS_MATCH_DETAIL"] = res.detail
 
            # Reorder columns: insert meta cols right after hs_col
            cols = list(df.columns)
            # remove meta cols from current order
            for mc in meta_cols:
                cols.remove(mc)
            hs_idx = cols.index(hs_col)
            new_cols = cols[:hs_idx+1] + meta_cols + cols[hs_idx+1:]
            df = df[new_cols]

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    out_buffer.seek(0)
    return out_buffer.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="HS Code Autofill", layout="wide")

st.title("HS Code Autofill (BDD + fuzzy + option web)")

st.markdown(
    """
**Objectif :** uploader un Excel avec une colonne HS Code vide → enrichir automatiquement depuis la BDD historique,  
avec un statut de matching + source.
"""
)

col1, col2 = st.columns(2)

with col1:
    input_file = st.file_uploader("1) Excel à compléter (multi-onglets OK)", type=["xlsx", "xlsm"])

with col2:
    bdd_file = st.file_uploader("2) BDD source de vérité (xlsm/xlsx)", type=["xlsx", "xlsm"])

st.divider()

fuzzy_threshold = st.slider("Seuil fuzzy (token_sort_ratio)", min_value=60, max_value=98, value=90, step=1)
enable_web = st.checkbox("Activer la recherche web (plus lent, best-effort)", value=False)

if input_file and bdd_file:
    st.success("Fichiers chargés. Tu peux lancer l’enrichissement.")

    if st.button("🚀 Enrichir et générer l’Excel", type="primary"):
        with st.spinner("Traitement en cours…"):
            out_bytes = process_workbook(
                input_bytes=input_file.read(),
                bdd_bytes=bdd_file.read(),
                fuzzy_threshold=fuzzy_threshold,
                enable_web=enable_web
            )

        st.download_button(
            label="⬇️ Télécharger l’Excel enrichi",
            data=out_bytes,
            file_name="excel_hs_enrichi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Charge l’Excel à compléter + la BDD, puis lance le traitement.")