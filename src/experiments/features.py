import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.experiments.catalog import FEATURE_BLOCK_CATALOG
from src.experiments.config import FeatureSpec
from src.features.headline_parser import parse_headlines
from src.features.headlines import (
    build_company_session_features,
    build_headline_event_table,
    build_headline_features,
    build_session_text_features,
    score_company_relevance,
)
from src.features.price import build_price_features

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _filter_sessions(frame: pd.DataFrame, sessions) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_index = pd.Index(sessions).astype(int)
    return frame.loc[frame["session"].isin(session_index)].copy()


def _empty_feature_frame(sessions) -> pd.DataFrame:
    return pd.DataFrame(index=pd.Index(sorted(pd.Index(sessions).unique())))


def _prefix_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(columns={column: f"{prefix}__{column}" for column in frame.columns})


def _fill_mixed_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.copy()
    for column in filled.columns:
        if pd.api.types.is_numeric_dtype(filled[column]):
            filled[column] = filled[column].fillna(0.0)
        else:
            filled[column] = filled[column].fillna("")
    return filled


class BaseFeatureBlock:
    def fit(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> "BaseFeatureBlock":
        return self

    def transform(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> pd.DataFrame:
        raise NotImplementedError


class PriceFeatureBlock(BaseFeatureBlock):
    def transform(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> pd.DataFrame:
        session_bars = _filter_sessions(bars, sessions)
        if session_bars.empty:
            return _empty_feature_frame(sessions)
        return build_price_features(session_bars).reindex(sorted(pd.Index(sessions).unique())).fillna(0.0)


def _tail_return(close: np.ndarray, lookback: int) -> float:
    if len(close) <= lookback:
        return 0.0
    return float(close[-1] / close[-1 - lookback] - 1.0)


def _calculate_rsi(prices: np.ndarray, window: int = 14) -> float:
    if len(prices) <= window + 1:
        return 50.0

    deltas = np.diff(prices)
    seed = deltas[: window + 1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = np.inf if down == 0 else up / down
    values = [100.0 - 100.0 / (1.0 + rs)]

    for idx in range(window, len(prices) - 1):
        delta = deltas[idx]
        upval = delta if delta > 0 else 0.0
        downval = -delta if delta < 0 else 0.0
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        rs = np.inf if down == 0 else up / down
        values.append(100.0 - 100.0 / (1.0 + rs))

    return float(values[-1] if values else 50.0)


def _skew(values: np.ndarray) -> float:
    if len(values) <= 2:
        return 0.0
    centered = values - values.mean()
    std = values.std(ddof=0)
    if std <= 1e-12:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _kurtosis_excess(values: np.ndarray) -> float:
    if len(values) <= 3:
        return 0.0
    centered = values - values.mean()
    std = values.std(ddof=0)
    if std <= 1e-12:
        return 0.0
    return float(np.mean((centered / std) ** 4) - 3.0)


def _autocorr(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag or lag <= 0:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    left_std = left.std(ddof=0)
    right_std = right.std(ddof=0)
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _price_technical_features(session_bars: pd.DataFrame) -> dict[str, float]:
    close_prices = session_bars["close"].to_numpy(dtype=float)
    open_prices = session_bars["open"].to_numpy(dtype=float)
    high_prices = session_bars["high"].to_numpy(dtype=float)
    low_prices = session_bars["low"].to_numpy(dtype=float)
    returns = np.diff(close_prices) / np.clip(close_prices[:-1], 1e-6, None)

    ema_10 = float(pd.Series(close_prices).ewm(span=10).mean().iloc[-1]) if len(close_prices) else 0.0
    ema_30 = float(pd.Series(close_prices).ewm(span=30).mean().iloc[-1]) if len(close_prices) else 0.0

    fft_vals = np.abs(np.fft.rfft(close_prices - close_prices.mean())) if len(close_prices) else np.asarray([])
    if len(fft_vals) >= 3:
        fft_vals = fft_vals.copy()
        fft_vals[0] = 0.0
        top_indices = np.argsort(fft_vals)[-2:]
        max_amp = float(fft_vals[top_indices[1]])
        second_amp = float(fft_vals[top_indices[0]])
    else:
        max_amp = 0.0
        second_amp = 0.0

    green_bars = float(np.mean(close_prices > open_prices)) if len(close_prices) else 0.0
    avg_spread = float(np.mean((high_prices - low_prices) / np.clip(open_prices, 1e-6, None))) if len(open_prices) else 0.0

    return {
        "ret_last_5": _tail_return(close_prices, lookback=5),
        "ret_last_10": _tail_return(close_prices, lookback=10),
        "ret_last_20": _tail_return(close_prices, lookback=20),
        "ema_10_30_cross": float(ema_10 / max(ema_30, 1e-6) - 1.0) if ema_30 else 0.0,
        "price_vs_ema10": float(close_prices[-1] / max(ema_10, 1e-6) - 1.0) if ema_10 else 0.0,
        "rsi_14": _calculate_rsi(close_prices, window=14),
        "ret_skew": _skew(returns),
        "ret_kurt": _kurtosis_excess(returns),
        "autocorr_1": _autocorr(returns, lag=1),
        "autocorr_5": _autocorr(returns, lag=5),
        "fourier_max_freq_amp": max_amp,
        "fourier_2nd_freq_amp": second_amp,
        "green_bar_ratio": green_bars,
        "avg_bar_spread": avg_spread,
    }


class PriceTechnicalFeatureBlock(BaseFeatureBlock):
    def transform(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> pd.DataFrame:
        session_bars = _filter_sessions(bars, sessions).sort_values(["session", "bar_ix"])
        if session_bars.empty:
            return _empty_feature_frame(sessions)

        rows: list[dict[str, float]] = []
        for session, session_frame in session_bars.groupby("session", sort=True):
            rows.append({"session": int(session), **_price_technical_features(session_frame)})

        feature_frame = pd.DataFrame(rows).set_index("session").sort_index()
        return feature_frame.reindex(sorted(pd.Index(sessions).unique())).fillna(0.0)


class HeadlineParserFeatureBlock(BaseFeatureBlock):
    def transform(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> pd.DataFrame:
        session_headlines = _filter_sessions(headlines, sessions)
        return build_headline_features(session_headlines, sessions=sessions).fillna(0.0).sort_index()


def _tokenize(text: str, ngram_range: tuple[int, int]) -> list[str]:
    tokens = TOKEN_PATTERN.findall(str(text or "").lower())
    if not tokens:
        return []

    min_n, max_n = ngram_range
    terms: list[str] = []
    for n in range(min_n, max_n + 1):
        if n <= 0 or len(tokens) < n:
            continue
        if n == 1:
            terms.extend(tokens)
            continue
        for start in range(len(tokens) - n + 1):
            terms.append("_".join(tokens[start : start + n]))
    return terms


class _TfidfVectorizer:
    def __init__(
        self,
        *,
        min_df: int = 2,
        max_features: int = 256,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.min_df = int(min_df)
        self.max_features = int(max_features)
        self.ngram_range = ngram_range
        self.terms_: list[str] = []
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, documents: pd.Series) -> "_TfidfVectorizer":
        doc_freq: Counter[str] = Counter()
        term_freq: Counter[str] = Counter()
        n_docs = int(len(documents))

        for text in documents.fillna("").astype(str):
            terms = _tokenize(text, self.ngram_range)
            if not terms:
                continue
            term_freq.update(terms)
            doc_freq.update(set(terms))

        kept_terms = [term for term, df in doc_freq.items() if df >= self.min_df]
        kept_terms.sort(key=lambda term: (-term_freq[term], -doc_freq[term], term))
        if self.max_features > 0:
            kept_terms = kept_terms[: self.max_features]

        self.terms_ = kept_terms
        self.vocabulary_ = {term: idx for idx, term in enumerate(self.terms_)}
        self.idf_ = np.asarray(
            [math.log((1.0 + n_docs) / (1.0 + doc_freq[term])) + 1.0 for term in self.terms_],
            dtype=float,
        )
        return self

    def transform(self, documents: pd.Series) -> pd.DataFrame:
        if not self.terms_:
            return pd.DataFrame(index=documents.index)

        matrix = np.zeros((len(documents), len(self.terms_)), dtype=float)
        for row_ix, text in enumerate(documents.fillna("").astype(str)):
            counts = Counter(term for term in _tokenize(text, self.ngram_range) if term in self.vocabulary_)
            total = float(sum(counts.values()))
            if total <= 0.0:
                continue

            for term, count in counts.items():
                column_ix = self.vocabulary_[term]
                matrix[row_ix, column_ix] = (count / total) * self.idf_[column_ix]

            norm = float(np.linalg.norm(matrix[row_ix]))
            if norm > 0.0:
                matrix[row_ix] /= norm

        return pd.DataFrame(matrix, index=documents.index, columns=self.terms_)


def _resolve_text_series(parsed: pd.DataFrame, text_source: str) -> pd.Series:
    parsed["headline"] = parsed["headline"].fillna("").astype(str)
    parsed["body"] = parsed["body"].fillna("").astype(str)
    parsed["event_family"] = parsed["event_family"].fillna("").astype(str)
    parsed["company"] = parsed["company"].fillna("").astype(str)

    if text_source == "headline":
        return parsed["headline"].str.lower()
    if text_source == "body":
        return parsed["body"].str.lower()
    if text_source == "company_body":
        return (parsed["company"] + " " + parsed["body"]).str.lower().str.strip()
    if text_source == "event_family":
        return parsed["event_family"].str.lower()
    raise ValueError(
        f"Unsupported text_source={text_source!r}. Expected one of: headline, body, company_body, event_family."
    )


def _build_session_documents(headlines: pd.DataFrame, *, sessions, text_source: str) -> pd.Series:
    session_index = pd.Index(sorted(pd.Index(sessions).unique()))
    if headlines.empty:
        return pd.Series("", index=session_index, dtype="object")

    parsed = parse_headlines(headlines)
    text_series = _resolve_text_series(parsed, text_source=text_source)
    documents = (
        parsed.assign(_text=text_series)
        .groupby("session", sort=True)["_text"]
        .agg(lambda values: " ".join(text for text in values if str(text).strip()))
    )
    documents = documents.reindex(session_index).fillna("")
    return documents.astype("object")


@dataclass
class _DocumentFrameTfidfFeatureBlock(BaseFeatureBlock):
    min_df: int = 2
    max_features: int = 256
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        self.vectorizers_: dict[str, _TfidfVectorizer] = {}
        self.numeric_columns_: list[str] = []
        self.text_columns_: list[str] = []
        self.output_columns_: list[str] = []

    def _build_document_frame(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> "_DocumentFrameTfidfFeatureBlock":
        frame = _fill_mixed_feature_frame(self._build_document_frame(bars=bars, headlines=headlines, sessions=sessions))
        self.text_columns_ = [
            column for column in frame.columns if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column])
        ]
        self.numeric_columns_ = [column for column in frame.columns if column not in self.text_columns_]
        self.vectorizers_ = {}
        for column in self.text_columns_:
            vectorizer = _TfidfVectorizer(
                min_df=self.min_df,
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
            vectorizer.fit(frame[column].astype(str))
            self.vectorizers_[column] = vectorizer

        transformed = self._transform_frame(frame)
        self.output_columns_ = list(transformed.columns)
        return self

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric_frame = (
            frame.reindex(columns=self.numeric_columns_)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sort_index()
            if self.numeric_columns_
            else pd.DataFrame(index=frame.index)
        )

        parts: list[pd.DataFrame] = [numeric_frame]
        for column in self.text_columns_:
            series = frame[column].astype(str) if column in frame.columns else pd.Series("", index=frame.index, dtype="object")
            transformed = self.vectorizers_[column].transform(series).add_prefix(f"{column}__")
            parts.append(transformed)

        result = pd.concat(parts, axis=1).fillna(0.0).sort_index() if parts else pd.DataFrame(index=frame.index)
        if self.output_columns_:
            result = result.reindex(columns=self.output_columns_, fill_value=0.0)
        return result

    def transform(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> pd.DataFrame:
        frame = _fill_mixed_feature_frame(self._build_document_frame(bars=bars, headlines=headlines, sessions=sessions))
        return self._transform_frame(frame)


@dataclass
class HeadlineTfidfFeatureBlock(_DocumentFrameTfidfFeatureBlock):
    text_source: str = "body"
    include_numeric: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

    def _build_document_frame(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> pd.DataFrame:
        documents = _build_session_documents(headlines, sessions=sessions, text_source=self.text_source).to_frame("session_text")
        if self.include_numeric:
            documents = documents.join(build_headline_features(headlines, sessions=sessions), how="left")
        return _fill_mixed_feature_frame(documents).sort_index()


def _build_company_document_frame(
    headlines: pd.DataFrame,
    *,
    sessions,
    text_source: str,
    top_k: int,
    include_numeric: bool,
) -> pd.DataFrame:
    session_index = pd.Index(sorted(pd.Index(sessions).unique()))
    if headlines.empty:
        return _empty_feature_frame(sessions)

    events = build_headline_event_table(headlines)
    company_features = score_company_relevance(build_company_session_features(events))
    if company_features.empty:
        return _empty_feature_frame(sessions)

    text_values = _resolve_text_series(events.copy(), text_source=text_source)
    company_documents = (
        events.assign(_text=text_values)
        .groupby(["session", "company"], sort=True)["_text"]
        .agg(lambda values: " ".join(text for text in values if str(text).strip()))
        .rename("company_text")
    )
    ranked = (
        company_features.join(company_documents, how="left")
        .reset_index()
        .sort_values(
            ["session", "relevance_score", "headline_count", "company"],
            ascending=[True, False, False, True],
        )
        .copy()
    )
    ranked["slot"] = ranked.groupby("session", sort=False).cumcount() + 1

    parts: list[pd.DataFrame] = []
    if include_numeric:
        parts.append(build_session_text_features(company_features, sessions=sessions, top_k=top_k))

    for slot in range(1, top_k + 1):
        slot_frame = ranked.loc[ranked["slot"] == slot, ["session", "company_text"]].set_index("session")
        slot_frame.columns = [f"top{slot}_text"]
        parts.append(slot_frame)

    document_frame = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=session_index)
    document_frame = document_frame.reindex(session_index)
    return _fill_mixed_feature_frame(document_frame).sort_index()


@dataclass
class HeadlineCompanyTfidfFeatureBlock(_DocumentFrameTfidfFeatureBlock):
    text_source: str = "body"
    top_k: int = 2
    include_numeric: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

    def _build_document_frame(self, *, bars: pd.DataFrame, headlines: pd.DataFrame, sessions) -> pd.DataFrame:
        return _build_company_document_frame(
            headlines,
            sessions=sessions,
            text_source=self.text_source,
            top_k=self.top_k,
            include_numeric=self.include_numeric,
        )

def build_feature_block(spec: FeatureSpec) -> BaseFeatureBlock:
    if spec.name == "price":
        return PriceFeatureBlock()

    if spec.name == "price_technical":
        return PriceTechnicalFeatureBlock()

    if spec.name == "headline_parser":
        return HeadlineParserFeatureBlock()

    if spec.name == "headline_tfidf":
        ngram_max = int(spec.params.get("ngram_max", 2))
        return HeadlineTfidfFeatureBlock(
            text_source=str(spec.params.get("text_source", "body")),
            include_numeric=bool(spec.params.get("include_numeric", False)),
            min_df=int(spec.params.get("min_df", 2)),
            max_features=int(spec.params.get("max_features", 256)),
            ngram_range=(1, ngram_max),
        )

    if spec.name == "headline_company_tfidf":
        ngram_max = int(spec.params.get("ngram_max", 2))
        return HeadlineCompanyTfidfFeatureBlock(
            text_source=str(spec.params.get("text_source", "body")),
            top_k=int(spec.params.get("top_k", 2)),
            include_numeric=bool(spec.params.get("include_numeric", False)),
            min_df=int(spec.params.get("min_df", 2)),
            max_features=int(spec.params.get("max_features", 256)),
            ngram_range=(1, ngram_max),
        )

    available = ", ".join(sorted(FEATURE_BLOCK_CATALOG))
    raise ValueError(f"Unsupported feature block={spec.name!r}. Available options: {available}.")


def build_feature_matrix(
    feature_specs: tuple[FeatureSpec, ...],
    block_instances: list[BaseFeatureBlock],
    *,
    bars: pd.DataFrame,
    headlines: pd.DataFrame,
    sessions,
) -> pd.DataFrame:
    session_index = pd.Index(sorted(pd.Index(sessions).unique()))
    feature_parts: list[pd.DataFrame] = []

    for spec, block in zip(feature_specs, block_instances, strict=True):
        feature_frame = block.transform(bars=bars, headlines=headlines, sessions=session_index)
        feature_frame = feature_frame.reindex(session_index).fillna(0.0).sort_index()
        feature_parts.append(_prefix_columns(feature_frame, spec.resolved_alias))

    if not feature_parts:
        return pd.DataFrame(index=session_index)

    return pd.concat(feature_parts, axis=1).fillna(0.0).sort_index()
