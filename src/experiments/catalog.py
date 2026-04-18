FEATURE_BLOCK_CATALOG: dict[str, str] = {
    "price": "Session-level OHLC price statistics from the seen bars.",
    "price_technical": "Legacy-style technical price indicators such as RSI, EMA spreads, autocorrelation, and FFT amplitudes.",
    "headline_parser": "Deterministic numeric headline parser features aggregated by session.",
    "headline_sequence": "Headline timing and polarity-sequence features aggregated by session.",
    "headline_regime_context": "Pre-event price regime context features aggregated around headline timing.",
    "headline_tfidf": "Fold-fitted session-level TF-IDF features built from parsed headline text.",
    "headline_company_tfidf": "Fold-fitted company-aware TF-IDF using the top relevance-ranked companies per session.",
}

MODEL_CATALOG: dict[str, str] = {
    "linear": "Unregularized linear regression with residual-std uncertainty.",
    "ridge": "L2-regularized linear regression with residual-std uncertainty.",
    "weighted_linear": "Sample-weighted linear regression that emphasizes larger-magnitude training returns.",
    "weighted_ridge": "Sample-weighted ridge regression that emphasizes larger-magnitude training returns.",
}


def format_catalog(title: str, entries: dict[str, str]) -> str:
    lines = [title]
    for name in sorted(entries):
        lines.append(f"- {name}: {entries[name]}")
    return "\n".join(lines)


def format_full_catalog() -> str:
    return "\n\n".join(
        [
            format_catalog("Supported Feature Blocks", FEATURE_BLOCK_CATALOG),
            format_catalog("Supported Models", MODEL_CATALOG),
        ]
    )
