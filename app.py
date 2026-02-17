import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
import requests


# ============================================================
# Normalisation / Helpers
# ============================================================
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


def safe_str(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    return "" if s.lower() == "nan" else s


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


# ============================================================
# Niveau 1 IA : Catégorisation (keywords) + gating
# ============================================================
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    # Textile / vêtements
    "TEXTILE": [
        "POLO", "T SHIRT", "TSHIRT", "TEE SHIRT", "CHEMISE", "PANTALON", "JEAN", "SHORT",
        "VESTE", "SWEAT", "PULL", "MAILLOT", "ROBE", "JUPE", "BLOUSE", "MANTEAU",
        "HOMME", "FEMME", "ENFANT", "BEBE", "TAILLE", "COTON", "POLYESTER", "ELASTHAN",
        "TEXTILE", "TENUE", "UNIFORME", "COSTUME", "CAPUCHE", "GILET", "CHAUSSETTE",
        "CASQUETTE", "CHAPEAU", "BONNET", "ECHARPE", "GANT", "GANTS"
    ],
    # Sport / loisirs / équipements sportifs
    "SPORT": [
        "BALLE", "BALL", "RAQUETTE", "RACKET", "ARC", "FLECHE", "FLECHES", "TIR",
        "SPORT", "SPORTIF", "ARCHERY", "BOW", "ARROW", "BULLETS", "BALLON",
        "EQUIPEMENT", "EQUIPEMENTS", "ENTRAINEMENT", "FITNESS", "GYM", "BADMINTON"
    ],
    # Meubles / mobilier / aménagement
    "FURNITURE": [
        "TABLE", "TABLE BASSE", "CHAISE", "FAUTEUIL", "CANAPE", "ARMOIRE", "ETAGERE",
        "MEUBLE", "MOBILIER", "BUREAU", "LIT", "MATELAS", "SOMMIER", "TABOURET",
        "PARASOL", "TRANSAT", "BANC", "TAPIS", "PLANCHE", "DECOR", "DECORATION"
    ],
    # Cosmétique / maquillage
    "COSMETICS": [
        "MAQUILLAGE", "ROUGE", "LEVRES", "LIPSTICK", "PINCEAU", "PAILLETTES",
        "COSMETIQUE", "BEAUTE", "EPONGE", "FARD", "GLOSS", "MASCARA"
    ],
    # Sacs / bagagerie / pochettes
    "BAGS": [
        "SAC", "POCHETTE", "BAG", "BAGEL", "POUCH", "VALISE", "LUGGAGE"
    ],
    # Food / alimentaire (très important pour éviter les œufs)
    "FOOD": [
        "OEUF", "OEUFS", "VOLAILLE", "POISSON", "VIANDE", "FROMAGE", "LAIT", "BEURRE",
        "BOISSON", "VIN", "BIERE", "EAU", "CHOCOLAT", "SUCRE", "FARINE"
    ],
    # Papier / imprimerie / papeterie (au cas où)
    "PAPER": [
        "PAPIER", "CARTON", "IMPRIME", "ETIQUETTE", "STICKER", "BROCHURE", "LIVRET"
    ],
}


def categorize_text(text: str) -> str:
    """
    Heuristic category: choose the category with the most keyword hits.
    If tie or no hit -> OTHER
    """
    t = normalize_text(text)
    if not t:
        return "OTHER"

    # Make sure multiword tokens match correctly
    score: Dict[str, int] = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        cnt = 0
        for kw in kws:
            kw_n = normalize_text(kw)
            # whole word-ish match
            if f" {kw_n} " in f" {t} ":
                cnt += 2  # strong hit
            elif kw_n in t:
                cnt += 1  # weak hit
        score[cat] = cnt

    best_cat = max(score, key=lambda k: score[k])
    if score[best_cat] <= 0:
        return "OTHER"

    # If top is not clearly better, keep OTHER to avoid over-filtering
    sorted_scores = sorted(score.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] == sorted_scores[1]:
        return "OTHER"

    return best_cat


# Allowed HS "chapters" (2-digit) per category — coarse but very effective
# HS is 6 digits; chapter is first 2 digits
ALLOWED_HS_CHAPTERS: Dict[str, List[str]] = {
    "TEXTILE": [str(i).zfill(2) for i in range(50, 64)],  # textiles, apparel, etc.
    "SPORT": ["95"],                                      # toys/sports goods (very common)
    "FURNITURE": ["94"],                                  # furniture
    "COSMETICS": ["33"],                                  # essential oils, perfumes, cosmetics
    "BAGS": ["42"],                                       # travel goods, bags, etc.
    "FOOD": [str(i).zfill(2) for i in range(1, 25)],       # broad food chapters
    "PAPER": ["48", "49"],                                 # paper / printed
    "OTHER": []                                            # no restriction
}


def hs_chapter(hs: str) -> str:
    digits = re.sub(r"\D", "", safe_str(hs))
    return digits[:2] if len(digits) >= 2 else ""


def hs_allowed_for_category(hs: str, cat: str) -> bool:
    cat = cat or "OTHER"
    allowed = ALLOWED_HS_CHAPTERS.get(cat, [])
    if not allowed:
        return True  # no restriction
    ch = hs_chapter(hs)
    return ch in allowed


# ============================================================
# TarifDouanier API lookup (WEB) — suggestion only
# ============================================================
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

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"}

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


# ============================================================
# Matching engine
# ============================================================
@dataclass
class MatchResult:
    hs_code: Optional[str]  # only if we fill HS CODE
    match_type: str         # EXACT / FUZZY / REVIEW / WEB_REVIEW / WEB_BLOCKED / NOT_FOUND / FOUND_NO_HS_IN_BDD / ALREADY_PRESENT / REVIEW_BLOCKED
    source: str             # BDD:BDD_1 / BDD:BDD_2 / WEB_API / NONE / INPUT
    detail: str             # include suggested_hs=... for review/web


# ============================================================
# Load BDD(s) and build indexes
# ============================================================
@st.cache_data(show_spinner=False)
def load_bdd_single(bdd_file_bytes: bytes, file_label: str) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(bdd_file_bytes), sheet_name=0, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    code_col = find_col(df, ["CODE ARTICLE", "CODE_ARTICLE", "ARTICLE", "Code article", "Article"])
    desc_col = find_col(df, ["Description", "DESIGNATION", "LIBELLE", "Libellé", "Libelle"])
    hs_col = find_col(df, ["HS CODE", "HS", "HS_CODE", "HS COD", "HSCODE"])
    supp_col = find_col(df, ["Fournisseur", "Supplier", "FOURNISSEUR"])

    df["_CODE_N"] = df[code_col].astype(str).map(normalize_text) if code_col else ""
    df["_DESC_N"] = df[desc_col].astype(str).map(normalize_text) if desc_col else ""
    df["_SUPP_N"] = df[supp_col].astype(str).map(normalize_text) if supp_col else ""
    df["_HS"] = df[hs_col] if hs_col else ""

    # Category inferred from description (best signal)
    df["_CAT"] = df["_DESC_N"].astype(str).map(categorize_text)

    df["_SRC_FILE"] = file_label
    df = df.fillna("")
    return df


def load_and_merge_bdds(bdd1_bytes: bytes, bdd2_bytes: Optional[bytes], bdd2_enabled: bool) -> pd.DataFrame:
    df1 = load_bdd_single(bdd1_bytes, file_label="BDD_1")
    if bdd2_enabled and bdd2_bytes:
        df2 = load_bdd_single(bdd2_bytes, file_label="BDD_2")
        return pd.concat([df1, df2], ignore_index=True)
    return df1


def build_bdd_indexes(bdd: pd.DataFrame):
    """
    Global + supplier-scoped indexes, plus category-specific desc lists.

    We store:
      - code_to_best[code] = (hs, srcfile)
      - desc_to_best[desc] = (hs, srcfile, cat)
      - desc_list (all desc)
      - desc_list_by_cat[cat] = list(desc)

    Supplier-scoped analogs.
    """
    code_to_best: Dict[str, Tuple[str, str]] = {}
    desc_to_best: Dict[str, Tuple[str, str, str]] = {}
    desc_list: List[str] = []
    desc_list_by_cat: Dict[str, List[str]] = {}

    supp_code_to_best: Dict[str, Dict[str, Tuple[str, str]]] = {}
    supp_desc_to_best: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
    supp_desc_list: Dict[str, List[str]] = {}
    supp_desc_list_by_cat: Dict[str, Dict[str, List[str]]] = {}

    for _, row in bdd.iterrows():
        code = safe_str(row.get("_CODE_N", ""))
        desc = safe_str(row.get("_DESC_N", ""))
        supp = safe_str(row.get("_SUPP_N", ""))
        hs = safe_str(row.get("_HS", ""))
        src_file = safe_str(row.get("_SRC_FILE", "BDD"))
        cat = safe_str(row.get("_CAT", "OTHER")) or "OTHER"

        if code and hs and code not in code_to_best:
            code_to_best[code] = (hs, src_file)

        if desc:
            desc_list.append(desc)
            desc_list_by_cat.setdefault(cat, []).append(desc)
            if hs and desc not in desc_to_best:
                desc_to_best[desc] = (hs, src_file, cat)

        if supp:
            supp_code_to_best.setdefault(supp, {})
            supp_desc_to_best.setdefault(supp, {})
            supp_desc_list.setdefault(supp, [])
            supp_desc_list_by_cat.setdefault(supp, {})

            if code and hs and code not in supp_code_to_best[supp]:
                supp_code_to_best[supp][code] = (hs, src_file)

            if desc:
                supp_desc_list[supp].append(desc)
                supp_desc_list_by_cat[supp].setdefault(cat, []).append(desc)
                if hs and desc not in supp_desc_to_best[supp]:
                    supp_desc_to_best[supp][desc] = (hs, src_file, cat)

    return (
        code_to_best,
        desc_to_best,
        desc_list,
        desc_list_by_cat,
        supp_code_to_best,
        supp_desc_to_best,
        supp_desc_list,
        supp_desc_list_by_cat,
    )


# ============================================================
# Fuzzy helpers
# ============================================================
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


# ============================================================
# Core match
# ============================================================
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
    Order:
      1) EXACT code
      2) EXACT desc
      3) FUZZY with category gating (avoid POLO clothes vs POLO balls)
      4) WEB suggestion only, and blocked if HS chapter incompatible with category
    """
    (
        code_to_best,
        desc_to_best,
        desc_list,
        desc_list_by_cat,
        supp_code_to_best,
        supp_desc_to_best,
        supp_desc_list,
        supp_desc_list_by_cat
    ) = indexes

    code_n = normalize_text(article_code)
    lib_n = normalize_text(libelle)
    supp_n = normalize_text(fournisseur)

    input_cat = categorize_text(libelle)  # category inferred from label
    scoped_code = supp_code_to_best.get(supp_n) if supp_n else None
    scoped_desc = supp_desc_to_best.get(supp_n) if supp_n else None
    scoped_list = supp_desc_list.get(supp_n) if supp_n else None
    scoped_list_by_cat = supp_desc_list_by_cat.get(supp_n) if supp_n else None

    # 1) Exact by code (supplier first) — trusted
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

    # 2) Exact by description — trusted
    if lib_n:
        if scoped_desc and lib_n in scoped_desc:
            hs, srcfile, cat = scoped_desc[lib_n]
            return MatchResult(hs, "EXACT", f"BDD:{srcfile}", f"Description exact (supplier-scoped); cat={cat}")
        if lib_n in desc_to_best:
            hs, srcfile, cat = desc_to_best[lib_n]
            return MatchResult(hs, "EXACT", f"BDD:{srcfile}", f"Description exact; cat={cat}")

    # 3) Fuzzy with category gating
    if lib_n:
        # Choose list by category if possible
        choices_source = "global"
        choices = None
        used_cat_filter = False

        # Supplier-scoped (category)
        if scoped_list_by_cat and input_cat != "OTHER":
            cat_list = scoped_list_by_cat.get(input_cat, [])
            if len(cat_list) >= 25:
                choices = cat_list
                choices_source = "supplier-scoped"
                used_cat_filter = True

        # Global (category)
        if choices is None and input_cat != "OTHER":
            cat_list = desc_list_by_cat.get(input_cat, [])
            if len(cat_list) >= 50:  # global needs bigger pool
                choices = cat_list
                choices_source = "global"
                used_cat_filter = True

        # Fallback (no category list large enough)
        if choices is None:
            if scoped_list and len(scoped_list) >= 25:
                choices = scoped_list
                choices_source = "supplier-scoped"
            else:
                choices = desc_list
                choices_source = "global"

        best, second = fuzzy_best_two(lib_n, choices)
        if best:
            best_desc, best_score = best
            second_score = second[1] if second else 0
            margin = best_score - second_score

            # Find HS for matched description
            hs_src: Optional[Tuple[str, str, str]] = None
            if choices_source == "supplier-scoped" and scoped_desc:
                hs_src = scoped_desc.get(best_desc)
            if not hs_src:
                hs_src = desc_to_best.get(best_desc)

            if hs_src:
                hs, srcfile, cand_cat = hs_src

                # If we didn't use category filter (fallback), block cross-category suggestions
                # This is the key anti "POLO" confusion guard
                if (not used_cat_filter) and (input_cat != "OTHER") and (cand_cat != input_cat):
                    return MatchResult(
                        None,
                        "REVIEW_BLOCKED",
                        f"BDD:{srcfile}",
                        f'blocked_by_category input_cat={input_cat} cand_cat={cand_cat}; '
                        f'suggested_hs={hs}; score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

                # Auto-fill only if very confident
                if best_score >= auto_fill_threshold and margin >= margin_top2:
                    # Extra safety: HS chapter should be compatible with category (if known)
                    if input_cat != "OTHER" and not hs_allowed_for_category(hs, input_cat):
                        return MatchResult(
                            None,
                            "REVIEW_BLOCKED",
                            f"BDD:{srcfile}",
                            f'blocked_by_hs_chapter input_cat={input_cat}; hs={hs}; '
                            f'score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                        )
                    return MatchResult(
                        hs,
                        "FUZZY",
                        f"BDD:{srcfile}",
                        f'fuzzy({choices_source}) AUTO; cat={input_cat}; hs={hs}; score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

                # REVIEW suggestion (no fill)
                if best_score >= review_threshold:
                    # even in review, show suggested_hs
                    return MatchResult(
                        None,
                        "REVIEW",
                        f"BDD:{srcfile}",
                        f'suggested_hs={hs}; fuzzy({choices_source}) REVIEW; input_cat={input_cat}; cand_cat={cand_cat}; '
                        f'score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

            else:
                # Matched desc but HS missing in BDD
                if enable_web and libelle:
                    web = lookup_hs_via_tarifdouanier_api(libelle, year=web_year, lang="fr")
                    if web:
                        hs_web, label = web
                        if input_cat != "OTHER" and not hs_allowed_for_category(hs_web, input_cat):
                            return MatchResult(
                                None,
                                "WEB_BLOCKED",
                                "WEB_API",
                                f'blocked_web_by_category input_cat={input_cat}; suggested_hs={hs_web}; label="{label[:120]}"; year={web_year}'
                            )
                        return MatchResult(
                            None,
                            "WEB_REVIEW",
                            "WEB_API",
                            f'suggested_hs={hs_web}; web(api) after fuzzy HS-missing; input_cat={input_cat}; fuzzy_score={best_score}; '
                            f'label="{label[:120]}"; year={web_year}'
                        )
                return MatchResult(
                    None,
                    "FOUND_NO_HS_IN_BDD",
                    "BDD",
                    f'fuzzy matched but HS missing in BDD; input_cat={input_cat}; score={best_score}; matched="{best_desc[:120]}"'
                )

    # 4) Web fallback — suggestion only (+ block if incompatible with category)
    if enable_web and libelle:
        web = lookup_hs_via_tarifdouanier_api(libelle, year=web_year, lang="fr")
        if web:
            hs_web, label = web
            if input_cat != "OTHER" and not hs_allowed_for_category(hs_web, input_cat):
                return MatchResult(
                    None,
                    "WEB_BLOCKED",
                    "WEB_API",
                    f'blocked_web_by_category input_cat={input_cat}; suggested_hs={hs_web}; label="{label[:120]}"; year={web_year}'
                )
            return MatchResult(
                None,
                "WEB_REVIEW",
                "WEB_API",
                f'suggested_hs={hs_web}; web(api) input_cat={input_cat}; "{label[:120]}"; year={web_year}'
            )

    return MatchResult(None, "NOT_FOUND", "NONE", f"no match; input_cat={input_cat}")


# ============================================================
# Process workbook (multi tabs)
# ============================================================
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

                # Fill only for EXACT / FUZZY (hs_code not None)
                hscode = safe_str(res.hs_code)
                if hscode:
                    df.at[i, hs_col] = hscode

                df.at[i, "HS_MATCH_TYPE"] = res.match_type
                df.at[i, "HS_SOURCE"] = res.source
                df.at[i, "HS_MATCH_DETAIL"] = res.detail

            # Reorder columns: meta cols right after HS col
            cols = list(df.columns)
            for mc in meta_cols:
                cols.remove(mc)
            hs_idx = cols.index(hs_col)
            new_cols = cols[:hs_idx + 1] + meta_cols + cols[hs_idx + 1:]
            df = df[new_cols]

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    out_buffer.seek(0)
    return out_buffer.getvalue()


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="HS Code Finder", layout="wide")
st.title("HS Code Finder — 2 BDD + fuzzy + IA niveau 1 (catégories anti-confusion) + web (safe suggestions)")

st.markdown(
    """
**But :** uploader un Excel avec une colonne HS Code vide → enrichir automatiquement depuis **1 ou 2 BDD** (sources de vérité).  
✅ **IA Niveau 1** : catégorisation par mots-clés (TEXTILE/SPORT/MEUBLE/...) pour éviter les confusions (`POLO homme` ≠ `balles de polo`).  
✅ **WEB safe** : la recherche web **ne remplit jamais** le HS. Elle propose (`WEB_REVIEW`) ou bloque (`WEB_BLOCKED`) si incohérent.
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    input_file = st.file_uploader("1) Excel à compléter (multi-onglets OK)", type=["xlsx", "xlsm"])
with col2:
    bdd1_file = st.file_uploader("2) BDD source de vérité #1 (xlsm/xlsx)", type=["xlsx", "xlsm"])
with col3:
    bdd2_file = st.file_uploader("3) BDD source de vérité #2 (optionnel)", type=["xlsx", "xlsm"])
bdd2_enabled = bdd2_file is not None

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    auto_fill_threshold = st.slider("Seuil AUTO-FILL (fuzzy) — remplit HS", 80, 99, 95, 1)
with c2:
    review_threshold = st.slider("Seuil REVIEW — suggestion sans remplir", 60, 98, 90, 1)
with c3:
    margin_top2 = st.slider("Marge Top1 vs Top2 (anti-rouge)", 0, 20, 8, 1)

enable_web = st.checkbox("Activer le web (tarifdouanier.eu) — suggestion only + blocage hors catégorie", value=True)
web_year = st.selectbox("Année tarif", options=[2024, 2025, 2026], index=2)

with st.expander("Voir les catégories & règles (debug)"):
    st.write("Catégories détectées via keywords :", list(CATEGORY_KEYWORDS.keys()) + ["OTHER"])
    st.write("Chapitres HS autorisés (2 digits) par catégorie :", ALLOWED_HS_CHAPTERS)
    st.caption("Si tu veux, on peut enrichir les keywords avec votre vocabulaire Club Med (très efficace).")

st.caption(
    "Conseil : pour limiter les erreurs → AUTO-FILL 95-97, REVIEW 88-92, marge 8-10. "
    "Les types WEB_BLOCKED / REVIEW_BLOCKED expliquent les blocages (catégorie/chapitre HS)."
)

if input_file and bdd1_file:
    st.success("Fichiers chargés. Lance l’enrichissement.")
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
    st.info("Charge l’Excel à compléter + au moins la BDD #1.")
