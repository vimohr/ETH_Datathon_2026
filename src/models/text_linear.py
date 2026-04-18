import math
import re
from collections import Counter

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _is_text_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


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


class _ColumnTfidfVectorizer:
    def __init__(
        self,
        *,
        prefix: str,
        min_df: int = 2,
        max_features: int = 256,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.prefix = prefix
        self.min_df = int(min_df)
        self.max_features = int(max_features)
        self.ngram_range = ngram_range
        self.terms_: list[str] = []
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, documents: pd.Series) -> "_ColumnTfidfVectorizer":
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

        column_names = [f"{self.prefix}{term}" for term in self.terms_]
        return pd.DataFrame(matrix, index=documents.index, columns=column_names)


class RidgeTextModel:
    def __init__(
        self,
        *,
        alpha: float = 10.0,
        text_columns: list[str] | None = None,
        min_df: int = 2,
        max_features: int = 256,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.alpha = float(alpha)
        self.text_columns = text_columns
        self.min_df = int(min_df)
        self.max_features = int(max_features)
        self.ngram_range = ngram_range

        self.numeric_columns_: list[str] = []
        self.text_columns_: list[str] = []
        self.vectorizers_: dict[str, _ColumnTfidfVectorizer] = {}
        self.feature_names_: list[str] = []
        self.intercept_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.residual_std_: float = 1.0

    def _infer_column_types(self, features: pd.DataFrame) -> tuple[list[str], list[str]]:
        if self.text_columns is None:
            text_columns = [column for column in features.columns if _is_text_dtype(features[column])]
        else:
            text_columns = [column for column in self.text_columns if column in features.columns]

        numeric_columns = [column for column in features.columns if column not in text_columns]
        return numeric_columns, text_columns

    def _numeric_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_columns_:
            return pd.DataFrame(index=features.index)
        numeric = features.reindex(columns=self.numeric_columns_).apply(pd.to_numeric, errors="coerce")
        return numeric.fillna(0.0).sort_index()

    def _transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        numeric = self._numeric_frame(features)
        parts: list[pd.DataFrame] = [numeric]

        for column in self.text_columns_:
            if column in features.columns:
                series = features[column]
            else:
                series = pd.Series("", index=features.index, dtype="object")
            part = self.vectorizers_[column].transform(series.astype(str))
            parts.append(part)

        transformed = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=features.index)
        if self.feature_names_:
            transformed = transformed.reindex(columns=self.feature_names_, fill_value=0.0)
        return transformed.fillna(0.0).sort_index()

    def fit(self, features: pd.DataFrame, target_return: pd.Series) -> "RidgeTextModel":
        frame = features.copy().sort_index()
        target = pd.Series(target_return, index=frame.index).astype(float)

        self.numeric_columns_, self.text_columns_ = self._infer_column_types(frame)
        self.vectorizers_ = {}
        for column in self.text_columns_:
            vectorizer = _ColumnTfidfVectorizer(
                prefix=f"{column}__",
                min_df=self.min_df,
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            ).fit(frame[column].astype(str))
            self.vectorizers_[column] = vectorizer

        transformed = self._transform_features(frame)
        self.feature_names_ = list(transformed.columns)

        design_matrix = np.column_stack([np.ones(len(transformed)), transformed.to_numpy(dtype=float)])
        penalty = np.eye(design_matrix.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        lhs = design_matrix.T @ design_matrix + self.alpha * penalty
        rhs = design_matrix.T @ target.to_numpy(dtype=float)
        coefficients = np.linalg.solve(lhs, rhs)

        self.intercept_ = float(coefficients[0])
        self.coef_ = coefficients[1:]

        residuals = target.to_numpy(dtype=float) - self.predict_expected_return(frame).to_numpy(dtype=float)
        residual_std = residuals.std(ddof=1) if len(residuals) > 1 else 0.0
        self.residual_std_ = float(max(residual_std, 1e-6))
        return self

    def predict_expected_return(self, features: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        transformed = self._transform_features(features)
        predictions = self.intercept_ + transformed.to_numpy(dtype=float) @ self.coef_
        return pd.Series(predictions, index=transformed.index, name="predicted_return")

    def predict_uncertainty(self, features: pd.DataFrame) -> pd.Series:
        transformed = self._transform_features(features)
        return pd.Series(
            np.full(len(transformed), self.residual_std_, dtype=float),
            index=transformed.index,
            name="predicted_uncertainty",
        )
