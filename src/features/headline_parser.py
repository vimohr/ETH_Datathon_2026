import re

import pandas as pd

TOPICS = [
    "enterprise software",
    "cloud infrastructure",
    "data analytics",
    "digital payments",
    "wireless connectivity",
    "supply chain optimization",
    "precision manufacturing",
    "process automation",
    "renewable storage",
    "automated logistics",
]

GEOGRAPHIES = [
    "North America",
    "Latin America",
    "Middle East",
    "Asia Pacific",
    "Central Europe",
    "Europe",
    "Scandinavia",
    "Southeast Asia",
]

COUNTERPARTIES = [
    "a global retailer",
    "a multinational manufacturer",
    "a national infrastructure agency",
    "a leading cloud platform",
    "a top-tier research institute",
    "a major logistics provider",
    "an international consortium",
]

EVENT_STARTERS = [
    "chief operating officer",
    "chief strategy officer",
    "to present",
    "to host",
    "secures",
    "reports",
    "sees",
    "announces",
    "faces",
    "files",
    "completes",
    "begins",
    "explores",
    "launches",
    "delays",
    "revises",
    "misses",
    "wins",
    "names",
    "recalls",
    "withdraws",
    "warns",
    "opens",
    "confirms",
    "raises",
    "loses",
    "enters",
    "achieves",
    "expands",
    "signs",
    "beats",
    "cfo",
]

EVENT_RULES = [
    ("class_action", "faces", -1, ("faces class action",)),
    ("regulatory_review", "faces", -1, ("faces regulatory review",)),
    ("regulatory_approval", "files", 1, ("files for regulatory approval",)),
    ("record_revenue", "reports", 1, ("record quarterly revenue",)),
    ("operating_income_decline", "reports", -1, ("decline in operating income",)),
    ("customer_acquisition_increase", "reports", 1, ("increase in customer acquisition",)),
    ("strong_demand", "reports", 1, ("reports strong demand",)),
    ("rising_costs", "reports", -1, ("reports rising costs pressuring margins",)),
    ("regional_revenue_decline", "reports", -1, ("reports unexpected decline in", "revenue")),
    ("margin_improvement", "sees", 1, ("margin improvement",)),
    ("new_customer_orders_drop", "sees", -1, ("drop in new customer orders",)),
    ("mixed_results", "sees", 0, ("mixed results",)),
    ("share_buyback", "announces", 1, ("share buyback program",)),
    ("breakthrough", "announces", 1, ("announces breakthrough",)),
    ("capex_plan", "announces", 0, ("capital expenditure plan",)),
    ("strategic_acquisition", "completes", 1, ("completes strategic acquisition",)),
    ("facility_upgrade", "completes", 0, ("completes planned facility upgrade",)),
    ("scheduled_maintenance", "begins", 0, ("begins scheduled maintenance",)),
    ("strategic_alternatives", "explores", 0, ("explores strategic alternatives",)),
    ("next_generation_launch", "launches", 1, ("launches next-generation",)),
    ("product_launch_delay", "delays", -1, ("delays product launch",)),
    ("strategy_revision", "revises", 0, ("revises long-term strategy",)),
    ("revenue_miss", "misses", -1, ("misses quarterly revenue estimates",)),
    ("industry_award", "wins", 1, ("wins industry award",)),
    ("new_head", "names", 0, ("names new head of",)),
    ("product_recall", "recalls", -1, ("recalls products in",)),
    ("market_withdrawal", "withdraws", -1, ("withdraws from",)),
    ("supply_chain_warning", "warns", -1, ("warns of supply chain disruptions",)),
    ("new_office", "opens", 0, ("opens new office in",)),
    ("conference_presentation", "to_present", 0, ("to present at",)),
    ("symposium_participation", "confirms", 0, ("confirms participation in",)),
    ("investor_day", "to_host", 0, ("to host investor day",)),
    ("executive_departure", "cfo", -1, ("steps down unexpectedly",)),
    ("executive_departure", "chief_operating_officer", -1, ("steps down unexpectedly",)),
    ("executive_open_letter", "cfo", -1, ("addresses investor concerns in open letter",)),
    ("executive_open_letter", "chief_strategy_officer", -1, ("addresses investor concerns in open letter",)),
    ("loses_key_contract", "loses", -1, ("loses key contract",)),
    ("guidance_raise", "raises", 1, ("raises full-year guidance",)),
    ("joint_venture", "enters", 1, ("enters joint venture",)),
    ("partnership", "signs", 1, ("signs multi-year partnership",)),
    ("regulatory_milestone", "achieves", 1, ("achieves key regulatory milestone",)),
    ("market_expansion", "expands", 1, ("expands operations into",)),
    ("contract_win", "secures", 1, ("secures", "contract with")),
    ("beats_expectations", "beats", 1, ("beats analyst expectations",)),
]

DEFAULT_POLARITY_BY_VERB = {
    "secures": 1,
    "reports": 0,
    "sees": 0,
    "announces": 0,
    "faces": -1,
    "files": 0,
    "completes": 0,
    "begins": 0,
    "explores": 0,
    "launches": 1,
    "delays": -1,
    "revises": 0,
    "misses": -1,
    "wins": 1,
    "names": 0,
    "recalls": -1,
    "withdraws": -1,
    "warns": -1,
    "opens": 0,
    "confirms": 0,
    "to_present": 0,
    "to_host": 0,
    "chief_operating_officer": -1,
    "chief_strategy_officer": -1,
    "cfo": -1,
    "raises": 1,
    "loses": -1,
    "enters": 1,
    "achieves": 1,
    "expands": 1,
    "signs": 1,
    "beats": 1,
}

AMOUNT_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)([MB])")
PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)%")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _extract_company_and_body(headline: str) -> tuple[str, str]:
    tokens = headline.split()
    max_company_tokens = min(4, max(len(tokens) - 1, 1))
    for split_ix in range(1, max_company_tokens + 1):
        body = " ".join(tokens[split_ix:])
        if any(body.lower().startswith(starter) for starter in EVENT_STARTERS):
            company = " ".join(tokens[:split_ix])
            return company, body

    company_tokens = tokens[:2] if len(tokens) >= 3 else tokens[:1]
    body_tokens = tokens[len(company_tokens) :]
    return " ".join(company_tokens), " ".join(body_tokens)


def _verb_family(body_lower: str) -> str:
    if body_lower.startswith("chief operating officer"):
        return "chief_operating_officer"
    if body_lower.startswith("chief strategy officer"):
        return "chief_strategy_officer"
    if body_lower.startswith("to present"):
        return "to_present"
    if body_lower.startswith("to host"):
        return "to_host"
    return body_lower.split()[0] if body_lower else "unknown"


def _parse_amount_m(text: str) -> float:
    match = AMOUNT_PATTERN.search(text)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    return value * 1000.0 if unit == "B" else value


def _parse_pct(text: str) -> float:
    match = PERCENT_PATTERN.search(text)
    return float(match.group(1)) if match else 0.0


def _first_match(text: str, candidates: list[str]) -> str:
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate.lower() in text.lower():
            return candidate
    return "none"


def _normalize_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = AMOUNT_PATTERN.sub(" amount_token ", normalized)
    normalized = PERCENT_PATTERN.sub(" pct_token ", normalized)
    tokens = TOKEN_PATTERN.findall(normalized)
    return " ".join(tokens)


def _classify_event(body_lower: str) -> tuple[str, str, int]:
    for event_family, verb_family, polarity, fragments in EVENT_RULES:
        if all(fragment in body_lower for fragment in fragments):
            return event_family, verb_family, polarity

    verb_family = _verb_family(body_lower)
    polarity = DEFAULT_POLARITY_BY_VERB.get(verb_family, 0)
    return f"{verb_family}_other", verb_family, polarity


def parse_headline(headline: str) -> dict[str, object]:
    text = str(headline or "").strip()
    company, body = _extract_company_and_body(text)
    body_lower = body.lower()

    event_family, verb_family, polarity = _classify_event(body_lower)
    amount_m = _parse_amount_m(text)
    pct = _parse_pct(text)
    topic = _first_match(text, TOPICS)
    geography = _first_match(text, GEOGRAPHIES)
    counterparty = _first_match(text, COUNTERPARTIES)

    return {
        "company": company,
        "body": body,
        "event_text": body,
        "headline_normalized": _normalize_text(text),
        "event_text_normalized": _normalize_text(body),
        "verb_family": verb_family,
        "event_family": event_family,
        "topic": topic,
        "geography": geography,
        "counterparty": counterparty,
        "amount_m": amount_m,
        "pct": pct,
        "polarity": int(polarity),
        "has_amount": int(amount_m > 0.0),
        "has_pct": int(pct > 0.0),
    }


def parse_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    if headlines.empty:
        return headlines.copy()

    parsed = headlines.copy()
    parsed["headline"] = parsed["headline"].fillna("")
    parsed_records = parsed["headline"].map(parse_headline).tolist()
    parsed_columns = pd.DataFrame(parsed_records, index=parsed.index)
    parsed = pd.concat([parsed, parsed_columns], axis=1)
    return parsed
