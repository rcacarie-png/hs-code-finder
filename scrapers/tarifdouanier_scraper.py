import csv
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.tarifdouanier.eu"
YEAR = 2026
START_URL = f"{BASE_URL}/{YEAR}/"
OUTPUT_CSV = f"data/raw/tarifdouanier_{YEAR}_raw.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HSCodeScraper/1.0; +internal project)"
}

SLEEP_SECONDS = 1.2
MAX_PAGES = 5000  # garde-fou


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def clean_hs_code(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    return digits


def is_same_domain(url: str) -> bool:
    return urlparse(url).netloc in {"tarifdouanier.eu", "www.tarifdouanier.eu"}


def is_year_page(url: str, year: int) -> bool:
    path = urlparse(url).path.strip("/")
    return path == str(year) or path.startswith(f"{year}/")


def extract_links(html: str, current_url: str, year: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(current_url, href)

        if not is_same_domain(full):
            continue
        if not is_year_page(full, year):
            continue

        # on évite les ancres / query parasites
        parsed = urlparse(full)
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        urls.append(clean)

    return list(dict.fromkeys(urls))


def extract_pairs_from_text(html: str) -> list[dict]:
    """
    Extraction heuristique basée sur le texte visible.
    Le site expose les codes dans le contenu texte des pages chapitres/résultats.
    On récupère les couples :
    - code 4/6/8 chiffres
    - libellé textuel proche
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [
        normalize_space(line)
        for line in soup.get_text("\n").splitlines()
    ]
    lines = [x for x in lines if x]

    results = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # ex: "0408", "040819", "04081900", "04 0819", "04 08", etc.
        code_digits = clean_hs_code(line)

        if len(code_digits) in {4, 6, 8, 10}:
            label = None

            # on cherche le premier texte pertinent dans les lignes suivantes
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j]
                nxt_digits = clean_hs_code(nxt)

                # on ignore les lignes trop proches d'un autre code
                if len(nxt_digits) in {4, 6, 8, 10}:
                    continue

                lowered = nxt.lower()
                if lowered in {"chapitre", "position", "sous-titre", "sous titre"}:
                    continue

                if len(nxt) >= 4:
                    label = nxt
                    break

            if label:
                results.append({
                    "hs_code_raw": line,
                    "hs_code": code_digits,
                    "label": label,
                })

        i += 1

    # dédoublonnage simple
    dedup = {}
    for row in results:
        key = (row["hs_code"], row["label"])
        dedup[key] = row

    return list(dedup.values())


def scrape_site(year: int):
    visited = set()
    queue = deque([f"{BASE_URL}/{year}/"])
    rows = []
    pages_count = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    while queue and pages_count < MAX_PAGES:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=25)
            if not resp.ok:
                print(f"[WARN] {resp.status_code} - {url}")
                continue
        except Exception as e:
            print(f"[ERROR] {url} -> {e}")
            continue

        pages_count += 1
        print(f"[{pages_count}] {url}")

        html = resp.text

        # extraction des paires code/libellé
        pairs = extract_pairs_from_text(html)
        for p in pairs:
            p["page_url"] = url
            p["year"] = year
            rows.append(p)

        # découverte de nouvelles pages
        for link in extract_links(html, url, year):
            if link not in visited:
                queue.append(link)

        time.sleep(SLEEP_SECONDS)

    return rows


def save_csv(rows: list[dict], path: str):
    fieldnames = ["year", "page_url", "hs_code_raw", "hs_code", "label"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows = scrape_site(YEAR)
    print(f"Rows extracted: {len(rows)}")
    save_csv(rows, OUTPUT_CSV)
    print(f"Saved to {OUTPUT_CSV}")