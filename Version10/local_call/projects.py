"""The 12 sellable Bengaluru projects, loaded from the spreadsheet."""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

BROCHURE_DIR = Path(__file__).resolve().parents[2] / "Brochure"
WORKBOOK = BROCHURE_DIR / "Sobha_Functional_Projects_data.xlsx"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

OUTSIDE_PROJECTS = (
    "oakshire",
    "dream gardens",
    "victoria park",
    "athena",
    "bellevue",
    "rivana",
    "whispering hill",
    "inizio",
    "strada",
    "aurum",
    "altus",
    "mountain mist",
    "elysia",
    "aranya",
    "waterfront",
    "arbor",
    "metropolis",
    "bela encosta",
    "atlantis",
    "evergreens",
    "silver estate",
    "marina one",
    "nesara",
)

AREA_QUESTION = (
    "Which part of Bengaluru are you considering — North, East, or South?"
)
CONFIGURATION_QUESTION = "May I know which configuration you are interested in?"
PURPOSE_QUESTION = "Are you looking at this to live in, or as an investment?"


def _shared_strings(book):
    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("m:si", NS):
        text = "".join(
            node.text or ""
            for node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        )
        values.append(text)
    return values


def _sheet_rows(book):
    strings = _shared_strings(book)
    sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall("m:sheetData/m:row", NS):
        cells = []
        for cell in row.findall("m:c", NS):
            kind = cell.attrib.get("t")
            node = cell.find("m:v", NS)
            value = "" if node is None or node.text is None else node.text
            if kind == "s":
                value = strings[int(value)]
            cells.append(value.replace("\n", " ").strip())
        if any(cells):
            rows.append(cells)
    return rows


def _spoken_name(name):
    cleaned = re.sub(r"\s+", " ", (name or "").strip())
    cleaned = re.sub(r"\bBangalore\b", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\bby The\b", "by the", cleaned)
    if cleaned.lower().startswith("sobha "):
        return "SOBHA " + cleaned.split(" ", 1)[1]
    return cleaned


def _aliases(name, spoken):
    raw = spoken.lower()
    found = {raw, raw.replace("sobha ", "")}
    found.add(re.sub(r"[^a-z0-9]+", " ", raw).strip())
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    found.add(compact)
    if "one world" in raw:
        found.add("oneworld")
    if "sacred grove" in raw:
        found.add("sacred grove")
    if "birdsong" in raw:
        found.add("chartered birdsong")
        found.add("birdsong")
    if "valley view" in raw:
        found.add("valley view")
    return tuple(sorted(item for item in found if item))


def _place(location):
    places = (
        ("electronic city", "Electronic City"),
        ("whitefield", "Whitefield"),
        ("panathur", "Panathur"),
        ("sarjapur", "Sarjapur"),
        ("koramangala", "Koramangala"),
        ("bannerghatta", "Bannerghatta"),
        ("chikka tirupathi", "Chikka Tirupathi"),
        ("rajanukunte", "Rajanukunte"),
        ("banashankari", "Banashankari"),
        ("kannamangala", "Kannamangala"),
        ("bhoganahalli", "Bhoganahalli"),
    )
    low = (location or "").lower()
    for needle, label in places:
        if needle in low:
            return label
    return ""


def _home_and_beds(theme, configuration):
    text = f"{theme} {configuration}".lower()
    if "plot" in text:
        return "plot", ()
    if "row house" in text:
        beds = _bedrooms(configuration)
        return "row_house", beds or (4,)
    return "apartment", _bedrooms(configuration)


def _bedrooms(configuration):
    found = []
    for match in re.finditer(r"\d(?:\.\d)?", configuration or ""):
        number = float(match.group(0))
        if number not in found:
            found.append(number)
    return tuple(found)


def _price(text):
    match = re.search(
        r"inr\s*₹?\s*(\d+(?:\.\d+)?)\s*(crore|lakh)",
        text or "",
        re.I,
    )
    if not match:
        return None, "", False
    amount = float(match.group(1))
    unit = match.group(2).lower()
    crore = amount / 100.0 if unit == "lakh" else amount
    shown = str(int(amount)) if amount == int(amount) else f"{amount:.2f}".rstrip("0").rstrip(".")
    spoken = f"INR {shown} {'Lakh' if unit == 'lakh' else 'Crore'}"
    return crore, spoken, True


def _row_to_project(cells):
    name, path, theme, combined, part, price_cell = cells[1:7]
    pieces = [piece.strip() for piece in combined.split("|")]
    location = pieces[0] if pieces else ""
    size_band = pieces[1] if len(pieces) > 1 else ""
    configuration = pieces[2] if len(pieces) > 2 else ""
    home, beds = _home_and_beds(theme, configuration)
    crore, spoken_price, priced = _price(price_cell)
    spoken = _spoken_name(name)
    return {
        "id": re.sub(r"[^a-z0-9]+", "-", spoken.lower()).strip("-"),
        "spoken": spoken,
        "aliases": _aliases(name, spoken),
        "brochure": Path(path).name,
        "theme": theme.strip(),
        "location": location,
        "place": _place(location),
        "size_band": size_band,
        "configuration": configuration,
        "home_type": home,
        "bedrooms": beds,
        "part": part.replace(" Bangalore", "").strip(),
        "price_crore": crore,
        "price_spoken": spoken_price,
        "price_known": priced,
    }


def load_projects():
    with zipfile.ZipFile(WORKBOOK) as book:
        rows = _sheet_rows(book)
    projects = [_row_to_project(row) for row in rows[1:] if len(row) >= 7]
    missing = [item["brochure"] for item in projects if not (BROCHURE_DIR / item["brochure"]).is_file()]
    if missing:
        raise FileNotFoundError("Missing brochures: " + ", ".join(missing))
    return projects


PROJECTS = load_projects()
BY_ID = {item["id"]: item for item in PROJECTS}


def townpark():
    return BY_ID["sobha-townpark"]


def brochure_path(project):
    return BROCHURE_DIR / project["brochure"]


def project_named_in(text):
    low = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    compact = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    found = []
    for project in PROJECTS:
        for alias in project["aliases"]:
            if not alias or alias in {"sobha", "the", "by", "lake"}:
                continue
            if " " in alias and alias in low:
                found.append(project)
                break
            if re.search(rf"\b{re.escape(alias)}\b", low):
                found.append(project)
                break
            if len(alias) > 6 and alias in compact:
                found.append(project)
                break
    unique = []
    for project in found:
        if project not in unique:
            unique.append(project)
    return unique


def outside_project_named(text):
    low = (text or "").lower()
    return any(name in low for name in OUTSIDE_PROJECTS)


def part_of_city(text):
    low = (text or "").lower()
    if re.search(r"\b(north|northern)\b", low):
        return "North"
    if re.search(r"\b(east|eastern)\b", low):
        return "East"
    if re.search(r"\b(south|southern)\b", low):
        return "South"
    return None


def place_named(text):
    low = (text or "").lower()
    for project in PROJECTS:
        if project["place"] and project["place"].lower() in low:
            return project["place"]
    return None


def bedrooms_named(text):
    low = (text or "").lower()
    if re.search(r"\b3\.5\b|three and a half|3 and a half", low):
        return 3.5
    words = (
        (1, r"\b(1|one)\s*-?\s*(bhk|bed|bedroom)"),
        (2, r"\b(2|two)\s*-?\s*(bhk|bed|bedroom)"),
        (3, r"\b(3|three)\s*-?\s*(bhk|bed|bedroom)"),
        (4, r"\b(4|four)\s*-?\s*(bhk|bed|bedroom)"),
    )
    for number, pattern in words:
        if re.search(pattern, low):
            return float(number)
    return None


def wants_a_plot(text):
    return re.search(r"\b(plot|plotted|villa plot)\b", text or "", re.I) is not None


def budget_ceiling(text):
    """A stated budget in crore, and whether the words mean strictly below it."""
    low = (text or "").lower().replace(",", "")
    amount = None
    crore = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|cr)\b", low)
    lakh = re.search(r"(\d+(?:\.\d+)?)\s*lakh", low)
    if crore:
        amount = float(crore.group(1))
    elif lakh:
        amount = float(lakh.group(1)) / 100.0
    if amount is None:
        return None, False
    strict = re.search(r"\b(below|under|less than)\b", low) is not None
    if re.search(r"\bor below\b|\bup to\b|\bupto\b|\bwithin\b", low):
        strict = False
    return amount, strict


def undecided(text):
    low = (text or "").lower()
    return re.search(
        r"\b(not sure|unsure|haven'?t decided|have not decided|not yet decided|"
        r"not decided|don'?t know|do not know|yet to decide|skip)\b",
        low,
    ) is not None


def _fits_bedrooms(project, bedrooms):
    beds = project["bedrooms"]
    if project["home_type"] == "plot" or not beds:
        return False
    if bedrooms == 3:
        return any(bed == 3 or bed == 3.5 for bed in beds)
    return any(abs(bed - bedrooms) < 0.01 for bed in beds)


def matching_projects(part=None, bedrooms=None, ceiling=None, strict=False, place=None, plots_only=False):
    chosen = []
    for project in PROJECTS:
        if part and project["part"] != part:
            continue
        if place and project["place"] != place:
            continue
        if plots_only and project["home_type"] != "plot":
            continue
        if bedrooms is not None and not _fits_bedrooms(project, bedrooms):
            continue
        if ceiling is not None:
            price = project["price_crore"]
            if price is None:
                continue
            if strict and price >= ceiling:
                continue
            if not strict and price > ceiling + 1e-9:
                continue
        chosen.append(project)
    return chosen


def only_plots(projects):
    return bool(projects) and all(project["home_type"] == "plot" for project in projects)


def _price_sentence(project):
    if not project["price_known"]:
        return f"The price for {project['spoken']} will be discussed by our sales advisor."
    return f"{project['spoken']} starts from {project['price_spoken']}."


def _lowest(projects, count):
    priced = [project for project in projects if project["price_crore"] is not None]
    priced.sort(key=lambda project: (project["price_crore"], project["spoken"]))
    return priced[:count]


def suggestion(projects, ceiling=None, strict=False):
    """The spoken shortlist. A long list keeps the two lowest starting prices."""
    if ceiling is not None and strict and ceiling <= 1:
        plots = [project for project in projects if project["home_type"] == "plot"]
        if len(plots) == 1:
            project = plots[0]
            return (
                f"{project['spoken']} is a plotted development in {project['part']} Bengaluru, "
                f"starting from {project['price_spoken']}. Shall I tell you about that?"
            )
    if len(projects) == 1:
        project = projects[0]
        where = project["place"] or project["location"]
        if project["price_known"]:
            return (
                f"{project['spoken']}, near {where}, starts from {project['price_spoken']}. "
                "Shall I tell you about that?"
            )
        return (
            f"{project['spoken']}, near {where}. "
            "The price will be discussed by our sales advisor. Shall I tell you about that?"
        )
    if len(projects) == 2:
        first, second = projects
        shared = first["place"] if first["place"] and first["place"] == second["place"] else ""
        where = f", both at {shared}" if shared else ""
        return (
            f"Two projects fit that: {first['spoken']} and {second['spoken']}{where}. "
            f"{_price_sentence(first)} {_price_sentence(second)} "
            "Which one shall I tell you about, or shall I cover both?"
        )
    places = []
    for project in projects:
        if project["place"] and project["place"] not in places:
            places.append(project["place"])
    if len(places) > 2:
        listed = ", ".join(places[:-1]) + f", or {places[-1]}"
        return f"A few projects fit that. Are you looking closer to {listed}?"
    lowest = _lowest(projects, 2)
    if len(lowest) == 2:
        return (
            f"Two of the closest are {lowest[0]['spoken']}, starting from {lowest[0]['price_spoken']}, "
            f"and {lowest[1]['spoken']}, starting from {lowest[1]['price_spoken']}. "
            "Our sales advisor can cover the others. Which one shall I tell you about, or shall I cover both?"
        )
    return "I will have our sales advisor get in touch with you. Have a good day."


def nearest_project(ceiling):
    priced = [project for project in PROJECTS if project["price_crore"] is not None]
    if not priced or ceiling is None:
        return None
    priced.sort(key=lambda project: (abs(project["price_crore"] - ceiling), project["price_crore"]))
    return priced[0]


def card(project):
    price = project["price_spoken"] if project["price_known"] else "to be discussed by the sales advisor"
    configuration = project["configuration"] or project["size_band"]
    return (
        f"Project: {project['spoken']}. Theme: {project['theme']}. "
        f"Location: {project['location']}. Part: {project['part']} Bengaluru. "
        f"Size: {project['size_band']}. Configuration: {configuration}. "
        f"Starting price: {price}."
    )
