import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
import requests


# ----------------------------
# Helpers
# ----------------------------
def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols:
            return cols[key]
    for c in df.columns:
        cl = str(c).lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def find_hs_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = str(c).lower().replace(" ", "")
        if "hs" in cl and "cod" in cl:
            return c
    return None


def safe_str(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    return "" if s.lower() == "nan" else s


# ----------------------------
# WEB sanity check (anti-absurde)
# ----------------------------
def is_food_like_label(label: str) -> bool:
    t = normalize_text(label)
    food_kw = [
        "OEUF", "OEUFS", "VOLAILLE", "POISSON", "VIANDE", "FROMAGE", "LAIT", "BEURRE",
        "BOISSON", "VIN", "BIERE", "EAU", "CHOCOLAT", "SUCRE", "FARINE", "RIZ", "PATES",
        "FRUIT", "LEGUME", "CONSERVE", "SAUCE", "EPICE", "THE", "CAFE"
    ]
    return any(k in t for k in food_kw)


def hs_chapter(hs: str) -> str:
    digits = re.sub(r"\D", "", safe_str(hs))
    return digits[:2] if len(digits) >= 2 else ""


def block_web_if_obviously_wrong(label: str, hs: str) -> bool:
    """
    Block only obvious wrong suggestions:
    - If suggested HS is in food chapters (01..24) but label doesn't look like food.
    """
    ch = hs_chapter(hs)
    if ch and ("01" <= ch <= "24") and (not is_food_like_label(label)):
        return True
    return False


# ----------------------------
# TarifDouanier API lookup (WEB)
# ----------------------------
def _extract_cn_candidates(api_data: Any) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    if api_data is None:
        return candidates

    if isinstance(api_data, dict):
        for key in ["items", "results", "data", "suggestions"]:
            if key in api_data and isinstance(api_data[key], list):
                api_data = api_data[key]
                break

    if not isinstance(api_data, list):
        return candidates

    for item in api_data:
        if not isinstance(item, dict):
            continue

        code = (
            item.get("cn")
            or item.get("code")
            or item.get("tariffNumber")
            or item.get("number")
            or item.get("value")
            or item.get("id")
        )

        label = (
            item.get("label")
            or item.get("text")
            or item.get("description")
            or item.get("desc")
            or item.get("name")
            or ""
        )

        if code:
            code_str = str(code).strip()
            code_digits = re.sub(r"\D", "", code_str)
            if len(code_digits) >= 6:
                candidates.append((code_digits[:6], str(label).strip()))
            else:
                if code_str.isdigit() and len(code_str) >= 6:
                    candidates.append((code_str[:6], str(label).strip()))
    return candidates


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def lookup_hs_via_tarifdouanier_api(term: str, year: int = 2024, lang: str = "fr") -> Optional[Tuple[str, str]]:
    term_n = normalize_text(term)
    if not term_n:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
    }

    try:
        url_v1 = "https://www.tarifdouanier.eu/api/v1/cnSuggest"
        r = requests.get(url_v1, params={"term": term_n, "lang": lang, "year": str(year)}, headers=headers, timeout=15)
        if r.ok:
            data = r.json()
            cands = _extract_cn_candidates(data)
            if cands:
                return cands[0]
    except Exception:
        pass

    try:
        url_v2 = "https://www.tarifdouanier.eu/api/v2/cnSuggest"
        r = requests.get(url_v2, params={"term": term_n, "lang": lang}, headers=headers, timeout=15)
        if r.ok:
            data = r.json()
            cands = _extract_cn_candidates(data)
            if cands:
                return cands[0]
    except Exception:
        pass

    return None


# ----------------------------
# Matching engine
# ----------------------------
@dataclass
class MatchResult:
    hs_code: Optional[str]   # For EXACT/FUZZY: filled. For REVIEW/WEB: suggested HS (now also written in HS column).
    match_type: str          # EXACT / FUZZY / REVIEW / WEB_REVIEW / WEB_BLOCKED / NOT_FOUND / FOUND_NO_HS_IN_BDD / ALREADY_PRESENT
    source: str              # BDD:BDD_1 / BDD:BDD_2 / WEB_API / NONE / INPUT
    detail: str              # explanation, label, scores, etc.


@st.cache_data(show_spinner=False)
def load_bdd_single(bdd_file_bytes: bytes, file_label: str) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(bdd_file_bytes), sheet_name=0, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    code_col = find_col(df, ["CODE ARTICLE", "CODE_ARTICLE", "ARTICLE", "Code article", "Article"])
    desc_col = find_col(df, ["Description", "DESIGNATION", "LIBELLE", "Libellé", "Libelle", "Libellé à imprimer", "Libelle a imprimer"])
    hs_col = find_col(df, ["HS CODE", "HS", "HS_CODE", "HS COD", "HSCODE"])
    supp_col = find_col(df, ["Fournisseur", "Supplier", "FOURNISSEUR"])

    df["_CODE_N"] = df[code_col].astype(str).map(normalize_text) if code_col else ""
    df["_DESC_N"] = df[desc_col].astype(str).map(normalize_text) if desc_col else ""
    df["_SUPP_N"] = df[supp_col].astype(str).map(normalize_text) if supp_col else ""
    df["_HS"] = df[hs_col] if hs_col else ""
    df["_SRC_FILE"] = file_label  # BDD_1 or BDD_2
    df = df.fillna("")
    return df


def load_and_merge_bdds(bdd1_bytes: bytes, bdd2_bytes: Optional[bytes], bdd2_enabled: bool) -> pd.DataFrame:
    df1 = load_bdd_single(bdd1_bytes, file_label="BDD_1")
    if bdd2_enabled and bdd2_bytes:
        df2 = load_bdd_single(bdd2_bytes, file_label="BDD_2")
        return pd.concat([df1, df2], ignore_index=True)
    return df1


def build_bdd_indexes(bdd: pd.DataFrame):
    code_to_best: Dict[str, Tuple[str, str]] = {}
    desc_to_best: Dict[str, Tuple[str, str]] = {}
    desc_list: List[str] = []

    supp_code_to_best: Dict[str, Dict[str, Tuple[str, str]]] = {}
    supp_desc_to_best: Dict[str, Dict[str, Tuple[str, str]]] = {}
    supp_desc_list: Dict[str, List[str]] = {}

    for _, row in bdd.iterrows():
        code = safe_str(row.get("_CODE_N", ""))
        desc = safe_str(row.get("_DESC_N", ""))
        supp = safe_str(row.get("_SUPP_N", ""))
        hs = safe_str(row.get("_HS", ""))
        src_file = safe_str(row.get("_SRC_FILE", "BDD"))

        if code and hs and code not in code_to_best:
            code_to_best[code] = (hs, src_file)

        if desc:
            desc_list.append(desc)
            if hs and desc not in desc_to_best:
                desc_to_best[desc] = (hs, src_file)

        if supp:
            supp_code_to_best.setdefault(supp, {})
            supp_desc_to_best.setdefault(supp, {})
            supp_desc_list.setdefault(supp, [])

            if code and hs and code not in supp_code_to_best[supp]:
                supp_code_to_best[supp][code] = (hs, src_file)

            if desc:
                supp_desc_list[supp].append(desc)
                if hs and desc not in supp_desc_to_best[supp]:
                    supp_desc_to_best[supp][desc] = (hs, src_file)

    return code_to_best, desc_to_best, desc_list, supp_code_to_best, supp_desc_to_best, supp_desc_list


def fuzzy_best_two(query: str, choices: List[str]):
    if not query or not choices:
        return None, None
    results = process.extract(query, choices, scorer=fuzz.WRatio, limit=2)
    if not results:
        return None, None
    best = results[0]
    second = results[1] if len(results) > 1 else None
    best_tuple = (best[0], int(best[1]))
    second_tuple = (second[0], int(second[1])) if second else None
    return best_tuple, second_tuple


def match_row(
    article_code: str,
    libelle: str,
    fournisseur: str,
    indexes,
    auto_fill_threshold: int,
    review_threshold: int,
    margin_top2: int,
    enable_web: bool,
    web_year: int
) -> MatchResult:
    """
    New rule:
      - In uncertain cases (REVIEW / WEB_REVIEW / WEB_BLOCKED), we ALSO return hs_code (suggested),
        so it will be written into the HS column to ease comparison.
      - Users must check HS_MATCH_TYPE to know if it's auto-filled or suggestion.
    """
    code_to_best, desc_to_best, desc_list, supp_code_to_best, supp_desc_to_best, supp_desc_list = indexes

    code_n = normalize_text(article_code)
    lib_n = normalize_text(libelle)
    supp_n = normalize_text(fournisseur)

    scoped_code = supp_code_to_best.get(supp_n) if supp_n else None
    scoped_desc = supp_desc_to_best.get(supp_n) if supp_n else None
    scoped_list = supp_desc_list.get(supp_n) if supp_n else None

    # 1) Exact by code
    if code_n:
        if scoped_code:
            best = scoped_code.get(code_n)
            if best:
                hs, srcfile = best
                return MatchResult(hs, "EXACT", f"BDD:{srcfile}", "CODE ARTICLE exact (supplier-scoped)")
        best = code_to_best.get(code_n)
        if best:
            hs, srcfile = best
            return MatchResult(hs, "EXACT", f"BDD:{srcfile}", "CODE ARTICLE exact")

    # 2) Exact by description
    if lib_n:
        if scoped_desc and lib_n in scoped_desc:
            hs, srcfile = scoped_desc[lib_n]
            return MatchResult(hs, "EXACT", f"BDD:{srcfile}", "Description exact (supplier-scoped)")
        if lib_n in desc_to_best:
            hs, srcfile = desc_to_best[lib_n]
            return MatchResult(hs, "EXACT", f"BDD:{srcfile}", "Description exact")

    # 3) Fuzzy
    if lib_n:
        choices_source = "global"
        if scoped_list and len(scoped_list) >= 25:
            choices = scoped_list
            choices_source = "supplier-scoped"
        else:
            choices = desc_list

        best, second = fuzzy_best_two(lib_n, choices)
        if best:
            best_desc, best_score = best
            second_score = second[1] if second else 0
            margin = best_score - second_score

            hs_src: Optional[Tuple[str, str]] = None
            if choices_source == "supplier-scoped" and scoped_desc:
                hs_src = scoped_desc.get(best_desc)
            if not hs_src:
                hs_src = desc_to_best.get(best_desc)

            if hs_src:
                hs, srcfile = hs_src

                # Auto-fill when confident
                if best_score >= auto_fill_threshold and margin >= margin_top2:
                    return MatchResult(
                        hs,
                        "FUZZY",
                        f"BDD:{srcfile}",
                        f'fuzzy({choices_source}) AUTO; score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

                # REVIEW (uncertain) -> return suggested HS so it is written in HS column
                if best_score >= review_threshold:
                    return MatchResult(
                        hs,
                        "REVIEW",
                        f"BDD:{srcfile}",
                        f'fuzzy({choices_source}) REVIEW; score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

            else:
                # Fuzzy matched but HS missing in BDD -> try web suggestion
                if enable_web and libelle:
                    web = lookup_hs_via_tarifdouanier_api(libelle, year=web_year, lang="fr")
                    if web:
                        hs_web, label = web
                        if block_web_if_obviously_wrong(libelle, hs_web):
                            return MatchResult(
                                hs_web,  # still write suggested HS in HS column
                                "WEB_BLOCKED",
                                "WEB_API",
                                f'web suggestion blocked (obvious mismatch); label="{label[:140]}"; year={web_year}'
                            )
                        return MatchResult(
                            hs_web,
                            "WEB_REVIEW",
                            "WEB_API",
                            f'web suggestion; label="{label[:140]}"; year={web_year}'
                        )
                return MatchResult(
                    None,
                    "FOUND_NO_HS_IN_BDD",
                    "BDD",
                    f'fuzzy matched but HS missing in BDD; score={best_score}; matched="{best_desc[:120]}"'
                )

    # 4) Web fallback
    if enable_web and libelle:
        web = lookup_hs_via_tarifdouanier_api(libelle, year=web_year, lang="fr")
        if web:
            hs_web, label = web
            if block_web_if_obviously_wrong(libelle, hs_web):
                return MatchResult(
                    hs_web,
                    "WEB_BLOCKED",
                    "WEB_API",
                    f'web suggestion blocked (obvious mismatch); label="{label[:140]}"; year={web_year}'
                )
            return MatchResult(
                hs_web,
                "WEB_REVIEW",
                "WEB_API",
                f'web suggestion; label="{label[:140]}"; year={web_year}'
            )

    return MatchResult(None, "NOT_FOUND", "NONE", "no match")


def process_workbook(
    input_bytes: bytes,
    bdd1_bytes: bytes,
    bdd2_bytes: Optional[bytes],
    bdd2_enabled: bool,
    auto_fill_threshold: int,
    review_threshold: int,
    margin_top2: int,
    enable_web: bool,
    web_year: int
) -> bytes:
    bdd = load_and_merge_bdds(bdd1_bytes, bdd2_bytes, bdd2_enabled)
    indexes = build_bdd_indexes(bdd)

    xls = pd.ExcelFile(io.BytesIO(input_bytes), engine="openpyxl")
    out_buffer = io.BytesIO()

    with pd.ExcelWriter(out_buffer, engine="openpyxl") as writer:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            hs_col = find_hs_col(df)
            if hs_col is None:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            article_col = find_col(df, ["Article", "CODE ARTICLE", "Code article"])
            lib_col = find_col(df, ["Libellé à imprimer", "Libelle a imprimer", "Description", "Libellé", "Libelle"])
            supp_col = find_col(df, ["Fournisseur", "Supplier", "FOURNISSEUR"])

            meta_cols = ["HS_MATCH_TYPE", "HS_SOURCE", "HS_MATCH_DETAIL"]
            for mc in meta_cols:
                if mc not in df.columns:
                    df[mc] = ""

            for i in range(len(df)):
                current_hs = safe_str(df.at[i, hs_col]) if i in df.index else ""
                if current_hs:
                    df.at[i, "HS_MATCH_TYPE"] = "ALREADY_PRESENT"
                    df.at[i, "HS_SOURCE"] = "INPUT"
                    df.at[i, "HS_MATCH_DETAIL"] = "HS already filled in input file"
                    continue

                article_val = safe_str(df.at[i, article_col]) if article_col else ""
                lib_val = safe_str(df.at[i, lib_col]) if lib_col else ""
                supp_val = safe_str(df.at[i, supp_col]) if supp_col else ""

                res = match_row(
                    article_code=article_val,
                    libelle=lib_val,
                    fournisseur=supp_val,
                    indexes=indexes,
                    auto_fill_threshold=auto_fill_threshold,
                    review_threshold=review_threshold,
                    margin_top2=margin_top2,
                    enable_web=enable_web,
                    web_year=web_year
                )

                # NEW RULE: write HS if res.hs_code exists (even for REVIEW/WEB*)
                hscode = safe_str(res.hs_code)
                if hscode:
                    df.at[i, hs_col] = hscode

                df.at[i, "HS_MATCH_TYPE"] = res.match_type
                df.at[i, "HS_SOURCE"] = res.source
                df.at[i, "HS_MATCH_DETAIL"] = res.detail

            # Reorder: meta cols right after HS col
            cols = list(df.columns)
            for mc in meta_cols:
                cols.remove(mc)
            hs_idx = cols.index(hs_col)
            new_cols = cols[:hs_idx + 1] + meta_cols + cols[hs_idx + 1:]
            df = df[new_cols]

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    out_buffer.seek(0)
    return out_buffer.getvalue()


# ----------------------------
# Streamlit UI (clean)
# ----------------------------
st.set_page_config(page_title="HS Code Finder", layout="wide")
st.title("HS Code Finder")

st.caption("Compléter automatiquement la colonne HS Code à partir d’1 ou 2 fichiers historiques, avec suggestions en cas d’incertitude.")

col1, col2, col3 = st.columns(3)
with col1:
    input_file = st.file_uploader("Excel à compléter (multi-onglets)", type=["xlsx", "xlsm"])
with col2:
    bdd1_file = st.file_uploader("BDD source #1", type=["xlsx", "xlsm"])
with col3:
    bdd2_file = st.file_uploader("BDD source #2 (optionnel)", type=["xlsx", "xlsm"])
bdd2_enabled = bdd2_file is not None

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    auto_fill_threshold = st.slider("Seuil AUTO-FILL (remplissage automatique)", 80, 99, 95, 1)
with c2:
    review_threshold = st.slider("Seuil REVIEW (suggestion)", 60, 98, 90, 1)
with c3:
    margin_top2 = st.slider("Marge Top1 vs Top2 (anti-faux positifs)", 0, 20, 8, 1)

enable_web = st.checkbox("Activer la recherche web (suggestions)", value=True)
web_year = st.selectbox("Année tarif", options=[2024, 2025, 2026], index=2)

st.caption(
    "Lecture des résultats : "
    "EXACT/FUZZY = fiable (rempli automatiquement). "
    "REVIEW/WEB_REVIEW/WEB_BLOCKED = HS écrit mais à valider. "
    "NOT_FOUND = aucune piste."
)

if input_file and bdd1_file:
    st.success("Fichiers chargés.")
    if st.button("🚀 Enrichir et générer l’Excel", type="primary"):
        with st.spinner("Traitement…"):
            out_bytes = process_workbook(
                input_bytes=input_file.read(),
                bdd1_bytes=bdd1_file.read(),
                bdd2_bytes=bdd2_file.read() if bdd2_enabled else None,
                bdd2_enabled=bdd2_enabled,
                auto_fill_threshold=auto_fill_threshold,
                review_threshold=review_threshold,
                margin_top2=margin_top2,
                enable_web=enable_web,
                web_year=web_year
            )

        st.download_button(
            label="⬇️ Télécharger l’Excel enrichi",
            data=out_bytes,
            file_name="excel_hs_enrichi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Charge l’Excel à compléter + au moins la BDD source #1.")
