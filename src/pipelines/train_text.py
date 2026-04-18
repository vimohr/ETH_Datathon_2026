import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.load import load_bars, load_headlines, load_train_bars
from src.data.targets import build_train_targets
from src.evaluation.validation import run_cross_validation
from src.features.headlines import build_headline_event_table, build_headline_features, build_text_feature_frame
from src.features.text_embeddings import build_embedding_feature_frame
from src.models.baseline import LinearBaselineModel
from src.models.text_linear import RidgeTextModel
from src.models.uncertainty import size_positions
from src.paths import OOF_DIR, SUBMISSIONS_DIR, ensure_output_dirs
from src.submission import build_submission, build_submission_metadata, save_submission


def _build_feature_frame(
    headlines: pd.DataFrame,
    *,
    bars: pd.DataFrame | None,
    sessions,
    feature_set: str,
    text_source: str,
    aggregation: str,
    top_k: int,
    include_structured: bool,
    embedding_model: str,
    embedding_batch_size: int,
    normalize_embeddings: bool,
    use_price_reactions: bool,
) -> pd.DataFrame:
    if feature_set == "parser":
        return build_headline_features(headlines, sessions=sessions).fillna(0.0).sort_index()

    if feature_set == "tfidf":
        return build_text_feature_frame(
            headlines,
            bars=bars if use_price_reactions else None,
            sessions=sessions,
            text_source=text_source,
            aggregation=aggregation,
            top_k=top_k,
            include_numeric=include_structured,
        ).sort_index()

    if feature_set == "embeddings":
        events = build_headline_event_table(headlines, bars=bars if use_price_reactions else None)
        return build_embedding_feature_frame(
            events,
            sessions=sessions,
            text_source=text_source,
            aggregation=aggregation,
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            normalize_embeddings=normalize_embeddings,
            top_k=top_k,
            include_structured=include_structured,
        ).sort_index()

    raise ValueError(f"Unsupported feature_set={feature_set!r}.")


def build_feature_matrix(
    split: str,
    *,
    feature_set: str,
    text_source: str,
    aggregation: str,
    top_k: int,
    include_structured: bool,
    embedding_model: str,
    embedding_batch_size: int,
    normalize_embeddings: bool,
    use_price_reactions: bool,
) -> pd.DataFrame:
    bars = load_bars(split, "seen")
    sessions = bars["session"].unique()
    headlines = load_headlines(split, "seen")
    return _build_feature_frame(
        headlines,
        bars=bars,
        sessions=sessions,
        feature_set=feature_set,
        text_source=text_source,
        aggregation=aggregation,
        top_k=top_k,
        include_structured=include_structured,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
        use_price_reactions=use_price_reactions,
    )


def build_model_factory(
    *,
    feature_set: str,
    min_df: int,
    max_features: int,
    ngram_max: int,
    ridge_alpha: float,
):
    if feature_set == "parser":
        return LinearBaselineModel

    return lambda: RidgeTextModel(
        alpha=ridge_alpha,
        min_df=min_df,
        max_features=max_features,
        ngram_range=(1, ngram_max),
    )


def build_model_name(
    *,
    feature_set: str,
    text_source: str,
    aggregation: str,
    include_structured: bool,
    embedding_model: str,
    use_price_reactions: bool,
) -> str:
    if feature_set == "parser":
        return "text_only_parser"

    if feature_set == "embeddings":
        model_slug = embedding_model.rsplit("/", maxsplit=1)[-1].lower().replace("-", "_")
        name_parts = ["text_embed", aggregation, text_source, model_slug]
        if use_price_reactions:
            name_parts.append("price_react")
        if include_structured:
            name_parts.append("structured")
        return "_".join(name_parts)

    name_parts = ["text_tfidf", aggregation, text_source]
    if use_price_reactions:
        name_parts.append("price_react")
    if include_structured:
        name_parts.append("structured")
    return "_".join(name_parts)


def run_pipeline(
    test_split: str,
    *,
    feature_set: str,
    text_source: str,
    aggregation: str,
    top_k: int,
    include_structured: bool,
    min_df: int,
    max_features: int,
    ngram_max: int,
    ridge_alpha: float,
    embedding_model: str,
    embedding_batch_size: int,
    normalize_embeddings: bool,
    use_price_reactions: bool,
    output_path=None,
    cv_only: bool = False,
):
    ensure_output_dirs()

    train_seen_bars, train_unseen_bars = load_train_bars()
    train_targets = build_train_targets(train_seen_bars, train_unseen_bars)
    train_headlines = load_headlines("train", "seen")
    train_features = _build_feature_frame(
        train_headlines,
        bars=train_seen_bars,
        sessions=train_targets.index,
        feature_set=feature_set,
        text_source=text_source,
        aggregation=aggregation,
        top_k=top_k,
        include_structured=include_structured,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
        use_price_reactions=use_price_reactions,
    )
    target_return = train_targets["target_return"].sort_index()

    model_factory = build_model_factory(
        feature_set=feature_set,
        min_df=min_df,
        max_features=max_features,
        ngram_max=ngram_max,
        ridge_alpha=ridge_alpha,
    )
    fold_summary, oof_predictions = run_cross_validation(
        train_features,
        target_return,
        model_factory=model_factory,
    )
    print(fold_summary.to_string(index=False))
    print(f"Mean CV Sharpe: {fold_summary['sharpe'].mean():.4f}")

    model_name = build_model_name(
        feature_set=feature_set,
        text_source=text_source,
        aggregation=aggregation,
        include_structured=include_structured,
        embedding_model=embedding_model,
        use_price_reactions=use_price_reactions,
    )
    oof_path = OOF_DIR / f"{model_name}_oof.csv"
    oof_predictions.to_csv(oof_path, index_label="session")
    print(f"Saved OOF predictions to {oof_path}")

    if cv_only:
        return None

    model = model_factory().fit(train_features, target_return)
    feature_count = len(getattr(model, "feature_names_", list(train_features.columns)))
    test_features = build_feature_matrix(
        test_split,
        feature_set=feature_set,
        text_source=text_source,
        aggregation=aggregation,
        top_k=top_k,
        include_structured=include_structured,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
        use_price_reactions=use_price_reactions,
    )
    predicted_return = model.predict_expected_return(test_features)
    predicted_uncertainty = model.predict_uncertainty(test_features)
    target_position = size_positions(predicted_return, predicted_uncertainty)

    submission = build_submission(test_features.index, target_position)
    expected_sessions = load_bars(test_split, "seen")["session"].unique()

    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = SUBMISSIONS_DIR / f"{timestamp}_{model_name}_{test_split}.csv"
    output_path = Path(output_path)
    metadata = build_submission_metadata(
        model_name=model_name,
        test_split=test_split,
        feature_count=int(feature_count),
        mean_cv_sharpe=float(fold_summary["sharpe"].mean()),
        include_headlines=True,
        notes=(
            f"feature_set={feature_set}; text_source={text_source}; aggregation={aggregation}; "
            f"include_structured={include_structured}; min_df={min_df}; max_features={max_features}; "
            f"ngram_max={ngram_max}; ridge_alpha={ridge_alpha}; "
            f"embedding_model={embedding_model}; embedding_batch_size={embedding_batch_size}; "
            f"normalize_embeddings={normalize_embeddings}; use_price_reactions={use_price_reactions}"
        ),
    )
    saved_path = save_submission(
        submission,
        output_path,
        expected_sessions=expected_sessions,
        metadata=metadata,
        latest_alias=f"latest_{test_split}.csv",
    )
    print(f"Saved submission to {saved_path}")
    return submission


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-only model and create a submission.")
    parser.add_argument(
        "--test-split",
        choices=["public_test", "private_test"],
        default="public_test",
        help="Which test split to score and export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Submission output path. Defaults to outputs/submissions/...",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run cross-validation and write OOF predictions without fitting on all data.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["parser", "tfidf", "embeddings"],
        default="parser",
        help="Which text feature family to use.",
    )
    parser.add_argument(
        "--text-source",
        choices=["headline", "event", "headline_normalized", "event_normalized"],
        default="event_normalized",
        help="Which parsed text field to aggregate for TF-IDF experiments.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["session", "company_top2", "company_topk", "company_weighted"],
        default="session",
        help="How to aggregate headlines into session-level text documents.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="How many company slots to keep when aggregation=company_topk.",
    )
    parser.add_argument(
        "--include-structured",
        action="store_true",
        help="Join parser-derived numeric headline features into the TF-IDF frame.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=256,
        help="Maximum TF-IDF terms per text column.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Largest n-gram size to include in TF-IDF features.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="L2 regularization strength for the TF-IDF ridge model.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for frozen embedding experiments.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size used when encoding text embeddings.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable L2 normalization inside the frozen embedding encoder.",
    )
    parser.add_argument(
        "--use-price-reactions",
        action="store_true",
        help="Use seen-bar short-horizon price reactions to improve company relevance weighting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        test_split=args.test_split,
        feature_set=args.feature_set,
        text_source=args.text_source,
        aggregation=args.aggregation,
        top_k=args.top_k,
        include_structured=args.include_structured,
        min_df=args.min_df,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        ridge_alpha=args.ridge_alpha,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        normalize_embeddings=not args.no_normalize_embeddings,
        use_price_reactions=args.use_price_reactions,
        output_path=args.output,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
