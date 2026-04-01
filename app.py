import io
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz


# ----------------------------
# Configuration
# ----------------------------
BDD_CLUBMED_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "clubmed_bdd.xlsx")
TARIF_DOUANIER_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "tarifdouanier_2026_clean.csv")

# Seuils pour fuzzy matching contre tarif douanier (style différent des descriptions Club Med)
TARIF_AUTO_THRESHOLD = 85
TARIF_REVIEW_THRESHOLD = 70

# Seuils pour fuzzy matching BDD Club Med
AUTO_FILL_THRESHOLD = 95
REVIEW_THRESHOLD = 90
MARGIN_TOP2 = 8

# GitHub configuration
GITHUB_OWNER = "rcacarie-png"
GITHUB_REPO = "hs-code-finder"
GITHUB_BDD_PATH = "data/processed/clubmed_bdd.xlsx"


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


def safe_str(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    return "" if s.lower() == "nan" else s


def get_ext(filename: str) -> str:
    filename = filename or ""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower().strip()


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


# ----------------------------
# "Vêtements" detection (simple tightening)
# ----------------------------
def is_apparel_label(label: str) -> bool:
    t = normalize_text(label)

    apparel_kw = [
        "POLO", "T SHIRT", "TSHIRT", "TEE SHIRT", "TEE", "CHEMISE", "BLOUSE",
        "PANTALON", "JEAN", "SHORT", "ROBE", "JUPE", "VESTE", "MANTEAU",
        "SWEAT", "HOODIE", "PULL", "CARDIGAN", "MAILLOT", "TENUE",
        "CHAUSSETTE", "CHAUSSETTES", "SOUS VETEMENT", "SOUSVETEMENT",
        "CASQUETTE", "BONNET", "ECHARPE", "GANT", "GANTS",
        "BASKET", "CHAUSSURE", "CHAUSSURES", "SANDALE", "SANDALES",
        "COTON", "POLYESTER", "ELASTHANNE", "LAINE", "LIN", "VISCOSE",
        "TAILLE", "HOMME", "FEMME", "ENFANT"
    ]

    return any(k in t for k in apparel_kw)



# ----------------------------
# Matching engine
# ----------------------------
@dataclass
class MatchResult:
    hs_code: Optional[str]
    match_type: str
    source: str
    detail: str


@st.cache_data(show_spinner=False)
def load_bdd_single(bdd_file_bytes: bytes, file_label: str) -> pd.DataFrame:
    # BDD expected in xlsx/xlsm mostly; openpyxl handles both.
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


@st.cache_data(show_spinner=False)
def load_bdd_bundled(path: str) -> Optional[pd.DataFrame]:
    """Load bundled BDD from local file (Club Med database).
    
    Supports .xlsx and .xlsm files via openpyxl engine.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return load_bdd_single(f.read(), file_label="BDD_CLUBMED")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_tarif_douanier(path: str) -> Optional[pd.DataFrame]:
    """Load bundled tariff database (CSV format)."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str)
        if df is not None and not df.empty:
            # Normaliser les colonnes pour l'indexation
            df["_DESC_N"] = df["DESCRIPTION"].map(normalize_text)
            df["_HS"] = df["HS_CODE"].map(lambda x: re.sub(r"\D", "", str(x))[:6])
            
            # Filtrer les lignes avec HS codes valides (au moins 6 chiffres)
            df = df[df["_HS"].str.len() >= 6].copy()
            
        return df if not df.empty else None
    except Exception:
        return None


def load_tarif_with_indexes(path: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]], Optional[List[str]]]:
    """Load tariff database and build indexes."""
    df = load_tarif_douanier(path)
    if df is None:
        return None, None, None
    
    desc_to_hs, desc_list = build_tarif_indexes(df)
    return df, desc_to_hs, desc_list


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


def build_tarif_indexes(tarif_df: pd.DataFrame):
    """Build indexes for tariff database using normalized columns."""
    desc_to_hs: Dict[str, str] = {}
    desc_list: List[str] = []

    for _, row in tarif_df.iterrows():
        desc = safe_str(row.get("_DESC_N", ""))
        hs = safe_str(row.get("_HS", ""))

        if desc and hs:
            desc_list.append(desc)
            if desc not in desc_to_hs:
                desc_to_hs[desc] = hs

    return desc_to_hs, desc_list


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
    tarif_desc_to_hs: Optional[Dict[str, str]],
    tarif_desc_list: Optional[List[str]],
    auto_fill_threshold: int,
    review_threshold: int,
    margin_top2: int
) -> MatchResult:
    code_to_best, desc_to_best, desc_list, supp_code_to_best, supp_desc_to_best, supp_desc_list = indexes

    code_n = normalize_text(article_code)
    lib_n = normalize_text(libelle)
    supp_n = normalize_text(fournisseur)

    scoped_code = supp_code_to_best.get(supp_n) if supp_n else None
    scoped_desc = supp_desc_to_best.get(supp_n) if supp_n else None
    scoped_list = supp_desc_list.get(supp_n) if supp_n else None

    # Tightening for apparel labels (reduce wrong matches)
    af = auto_fill_threshold
    rv = review_threshold
    mg = margin_top2
    if is_apparel_label(libelle):
        af = min(99, af + 2)
        rv = min(98, rv + 2)
        mg = mg + 2

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
                if best_score >= af and margin >= mg:
                    return MatchResult(
                        hs,
                        "FUZZY",
                        f"BDD:{srcfile}",
                        f'fuzzy({choices_source}) AUTO; score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

                # REVIEW (uncertain) -> still write suggested HS in HS column
                if best_score >= rv:
                    return MatchResult(
                        hs,
                        "REVIEW",
                        f"BDD:{srcfile}",
                        f'fuzzy({choices_source}) REVIEW; score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

            else:
                # Fuzzy matched but HS missing in BDD
                return MatchResult(
                    None,
                    "FOUND_NO_HS_IN_BDD",
                    "BDD",
                    f'fuzzy matched but HS missing in BDD; score={best_score}; matched="{best_desc[:120]}"'
                )

    # 4) Fuzzy matching against Tarif Douanier if no match in BDD
    if lib_n and tarif_desc_list and tarif_desc_to_hs:
        best, second = fuzzy_best_two(lib_n, tarif_desc_list)
        if best:
            best_desc, best_score = best
            second_score = second[1] if second else 0
            margin = best_score - second_score

            hs = tarif_desc_to_hs.get(best_desc)
            if hs:
                # Auto-fill when confident (lower thresholds for tarif due to style differences)
                if best_score >= TARIF_AUTO_THRESHOLD and margin >= mg:
                    return MatchResult(
                        hs,
                        "FUZZY",
                        "TARIF",
                        f'fuzzy(tarif) AUTO; score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

                # REVIEW (uncertain) -> still write suggested HS in HS column
                # Require minimum margin of 5 to avoid false positives
                if best_score >= TARIF_REVIEW_THRESHOLD and margin >= 5:
                    return MatchResult(
                        hs,
                        "REVIEW",
                        "TARIF",
                        f'fuzzy(tarif) REVIEW; score={best_score}; top2_margin={margin}; matched="{best_desc[:120]}"'
                    )

    return MatchResult(None, "NOT_FOUND", "NONE", "no match")


def process_workbook(
    input_bytes: bytes,
    input_name: str,
    bdd: pd.DataFrame,
    tarif_desc_to_hs: Optional[Dict[str, str]],
    tarif_desc_list: Optional[List[str]],
    auto_fill_threshold: int,
    review_threshold: int,
    margin_top2: int
) -> bytes:
    # Use provided BDD
    indexes = build_bdd_indexes(bdd)

    # Choose engine for INPUT workbook based on extension
    ext = get_ext(input_name)
    if ext == "xls":
        # requires xlrd in requirements
        xls = pd.ExcelFile(io.BytesIO(input_bytes), engine="xlrd")
    else:
        # xlsx/xlsm
        xls = pd.ExcelFile(io.BytesIO(input_bytes), engine="openpyxl")

    out_buffer = io.BytesIO()

    # Output always xlsx
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
                # PATCH 1 : accès sécurisé à la valeur HS courante
                try:
                    current_hs = safe_str(df.at[i, hs_col])
                except (KeyError, IndexError):
                    current_hs = ""

                if current_hs:
                    df.at[i, "HS_MATCH_TYPE"] = "ALREADY_PRESENT"
                    df.at[i, "HS_SOURCE"] = "INPUT"
                    df.at[i, "HS_MATCH_DETAIL"] = "HS already filled in input file"
                    continue

                # PATCH 2 : accès sécurisé aux colonnes optionnelles
                try:
                    article_val = safe_str(df.at[i, article_col]) if article_col else ""
                except (KeyError, IndexError):
                    article_val = ""
                try:
                    lib_val = safe_str(df.at[i, lib_col]) if lib_col else ""
                except (KeyError, IndexError):
                    lib_val = ""
                try:
                    supp_val = safe_str(df.at[i, supp_col]) if supp_col else ""
                except (KeyError, IndexError):
                    supp_val = ""

                res = match_row(
                    article_code=article_val,
                    libelle=lib_val,
                    fournisseur=supp_val,
                    indexes=indexes,
                    tarif_desc_to_hs=tarif_desc_to_hs,
                    tarif_desc_list=tarif_desc_list,
                    auto_fill_threshold=auto_fill_threshold,
                    review_threshold=review_threshold,
                    margin_top2=margin_top2
                )

                # Write HS if suggested or exact
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

        # PATCH 3 : garantir qu'au moins un onglet est visible avant la sauvegarde
        # Corrige le IndexError "At least one sheet must be visible" sur openpyxl récent (Python 3.13)
        wb = writer.book
        visible_sheets = [ws for ws in wb.worksheets if ws.sheet_state == "visible"]
        if not visible_sheets and wb.worksheets:
            wb.worksheets[0].sheet_state = "visible"
        if wb.worksheets:
            wb.active = wb.worksheets[0]

    out_buffer.seek(0)
    return out_buffer.getvalue()


def enrich_and_push_bdd(enrich_file_bytes: bytes, enrich_file_name: str) -> int:
    """Enrich BDD with new lines from uploaded file and push to GitHub.
    Returns: number of new lines added, or -1 on error."""
    import base64
    import requests
    
    try:
        # Check for GitHub token
        if "GITHUB_TOKEN" not in st.secrets:
            st.warning("⚠️ GITHUB_TOKEN non configuré dans les secrets Streamlit.")
            return -1
        
        github_token = st.secrets["GITHUB_TOKEN"]
        
        # Read enriched file
        ext = get_ext(enrich_file_name)
        if ext == "xls":
            enrich_df = pd.read_excel(io.BytesIO(enrich_file_bytes), sheet_name=0, engine="xlrd")
        else:
            enrich_df = pd.read_excel(io.BytesIO(enrich_file_bytes), sheet_name=0, engine="openpyxl")
        
        enrich_df.columns = [str(c).strip() for c in enrich_df.columns]
        
        # 1. Détecter la colonne HS code
        hs_col = find_hs_col(enrich_df)
        if not hs_col:
            st.error("❌ Colonne HS CODE non trouvée dans le fichier.")
            return -1
        
        # 2. Détecter la colonne code article
        code_col = find_col(enrich_df, ["CODE ARTICLE", "Article", "CODE_ARTICLE"])
        if not code_col:
            st.warning("⚠️ Colonne CODE ARTICLE non trouvée, utilisation de valeurs vides.")
        
        # 3. Détecter la colonne description
        desc_col = find_col(enrich_df, ["Description", "Libellé à imprimer", "Libelle a imprimer", "DESIGNATION"])
        if not desc_col:
            st.warning("⚠️ Colonne Description non trouvée, utilisation de valeurs vides.")
        
        # 4. Garder uniquement les lignes où HS code est non vide
        enrich_df = enrich_df[
            enrich_df[hs_col].notna() & 
            (enrich_df[hs_col].astype(str).str.strip() != "") &
            (enrich_df[hs_col].astype(str).str.lower() != "nan")
        ].copy()
        
        if enrich_df.empty:
            st.warning("⚠️ Aucune ligne avec HS code rempli trouvée.")
            return 0
        
        # 5. Construire un DataFrame avec exactement ces colonnes pour correspondre à la BDD Club Med
        new_rows = []
        for _, row in enrich_df.iterrows():
            new_row = {
                "FOURNISSEUR": "",  # Laisser vide si colonne absente
                "CODE ARTICLE": safe_str(row.get(code_col, "")) if code_col else "",
                "Description": safe_str(row.get(desc_col, "")) if desc_col else "",
                "HS CODE": safe_str(row[hs_col])
            }
            new_rows.append(new_row)
        
        enrich_df_clean = pd.DataFrame(new_rows)
        
        # Charger la BDD existante
        existing_bdd = pd.read_excel(BDD_CLUBMED_PATH, sheet_name=0, engine="openpyxl")
        
        # 6. Concaténer avec la BDD existante et dédupliquer sur (CODE ARTICLE, HS CODE) en ignorant les valeurs vides
        combined_df = pd.concat([existing_bdd, enrich_df_clean], ignore_index=True)
        
        # Créer une clé de déduplication qui ignore les lignes où CODE ARTICLE ou HS CODE sont vides
        combined_df["_DEDUP_KEY"] = combined_df.apply(
            lambda row: f"{safe_str(row['CODE ARTICLE'])}|{safe_str(row['HS CODE'])}" 
            if safe_str(row['CODE ARTICLE']) and safe_str(row['HS CODE']) 
            else None, 
            axis=1
        )
        
        # Garder seulement les lignes avec une clé valide et dédupliquer
        combined_df = combined_df[combined_df["_DEDUP_KEY"].notna()]
        combined_df = combined_df.drop_duplicates(subset=["_DEDUP_KEY"], keep="first")
        combined_df = combined_df.drop(columns=["_DEDUP_KEY"])
        
        # Calculer le nombre de nouvelles lignes ajoutées
        new_count = len(enrich_df_clean)
        
        # Save to buffer
        out_buffer = io.BytesIO()
        with pd.ExcelWriter(out_buffer, engine="openpyxl") as writer:
            combined_df.to_excel(writer, sheet_name="BDD", index=False)
        out_buffer.seek(0)
        bdd_bytes = out_buffer.getvalue()
        
        # Get current file SHA
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{GITHUB_BDD_PATH}"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {github_token}",
        }
        
        resp_get = requests.get(url, headers=headers)
        if resp_get.status_code == 200:
            current_sha = resp_get.json().get("sha")
        elif resp_get.status_code == 404:
            current_sha = None
        else:
            st.error(f"❌ Erreur GitHub GET: {resp_get.status_code} - {resp_get.text}")
            return -1
        
        # Push updated file
        bdd_base64 = base64.b64encode(bdd_bytes).decode("utf-8")
        payload = {
            "message": f"BDD enrichie automatiquement - {new_count} nouv. lignes",
            "content": bdd_base64,
        }
        if current_sha:
            payload["sha"] = current_sha
        
        resp_put = requests.put(url, json=payload, headers=headers)
        if resp_put.status_code not in [200, 201]:
            st.error(f"❌ Erreur GitHub PUT: {resp_put.status_code} - {resp_put.text}")
            return -1
        
        # Clear cache for next load
        st.cache_data.clear()
        
        return max(0, new_count)
    
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return -1


# ----------------------------
# Streamlit UI (clean)
# ----------------------------
st.set_page_config(page_title="HS Code Finder", layout="wide")
st.title("HS Code Finder")
st.caption("Compléter automatiquement la colonne HS Code à partir de la BDD Club Med chargée.")

# Load bundled BDD at startup
bdd_clubmed = load_bdd_bundled(BDD_CLUBMED_PATH)
tarif_douanier, tarif_desc_to_hs, tarif_desc_list = load_tarif_with_indexes(TARIF_DOUANIER_PATH)

# Sidebar: Sources bundlées et ordre de priorité
with st.sidebar:
    st.header("📚 Sources de données")
    
    # BDD Club Med status
    st.subheader("1️⃣ BDD Club Med")
    if bdd_clubmed is not None and not bdd_clubmed.empty:
        st.success(f"✅ Chargée: {len(bdd_clubmed)} lignes")
    else:
        st.error(f"❌ Introuvable ou vide")
    
    # Tarif Douanier status
    st.subheader("2️⃣ Data tarifdouanier.eu")
    if tarif_douanier is not None and not tarif_douanier.empty:
        st.success(f"✅ Chargée: {len(tarif_douanier)} lignes")
    else:
        st.error(f"❌ Introuvable ou vide")
    
    st.divider()
    
    # Ordre de priorité
    st.subheader("🔍 Ordre de priorité")
    st.markdown("""
    1. **Code article exact** (BDD Club Med)
    2. **Description exacte** (BDD Club Med)
    3. **Fuzzy matching** (BDD Club Med)
       - AUTO si score >= seuil
       - REVIEW si incertain
    4. **Tarif fuzzy** (Data tarifdouanier.eu)
    5. **NOT_FOUND**
    """)

col1 = st.columns(1)[0]
with col1:
    input_file = st.file_uploader(
        "Excel à compléter (multi-onglets)",
        type=["xls", "xlsx", "xlsm"]
    )

st.divider()

st.caption(
    "Résultats : EXACT = code/description exacte (auto-rempli). "
    "FUZZY AUTO = match confiant (auto-rempli). "
    "REVIEW = match incertain à valider. "
    "TARIF_FUZZY = suggestion tarifaire à valider. "
    "NOT_FOUND = aucune correspondance."
)

if input_file and bdd_clubmed is not None and not bdd_clubmed.empty:
    st.success("Fichiers prêts.")
    if st.button("🚀 Enrichir et générer l'Excel", type="primary"):
        with st.spinner("Traitement…"):
            out_bytes = process_workbook(
                input_bytes=input_file.read(),
                input_name=input_file.name,
                bdd=bdd_clubmed,
                tarif_desc_to_hs=tarif_desc_to_hs,
                tarif_desc_list=tarif_desc_list,
                auto_fill_threshold=AUTO_FILL_THRESHOLD,
                review_threshold=REVIEW_THRESHOLD,
                margin_top2=MARGIN_TOP2
            )

        st.download_button(
            label="⬇️ Télécharger l'Excel enrichi",
            data=out_bytes,
            file_name="excel_hs_enrichi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
elif not input_file:
    st.info("Charge l'Excel à compléter.")
else:
    st.error("La BDD Club Med n'est pas chargée. Vérifiez que le fichier existe.")
# ----------------------------
# Section: Enrichir la BDD
# ----------------------------
st.divider()
st.subheader("📥 Enrichir la BDD")
st.caption("Uploadez un fichier Excel complété manuellement pour enrichir automatiquement la BDD Club Med.")

enrich_file = st.file_uploader(
    "Fichier Excel complété",
    type=["xls", "xlsx", "xlsm"],
    key="enrich_uploader"
)

if enrich_file:
    if st.button("Intégrer dans la BDD", type="primary"):
        with st.spinner("Intégration en cours…"):
            new_count = enrich_and_push_bdd(enrich_file.read(), enrich_file.name)
            if new_count > 0:
                st.success(f"✅ BDD enrichie : {new_count} nouvelles lignes ajoutées.")
                st.balloons()
                st.rerun()
            elif new_count == 0:
                st.info("✅ Aucune nouvelle ligne à ajouter (doublons détectés).")