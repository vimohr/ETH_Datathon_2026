# Experiment Log

| Timestamp | Owner | Features | Model | CV Sharpe | Public LB | Notes |
|-----------|-------|----------|-------|-----------|-----------|-------|
| 2026-04-18 | Codex | parser baseline headline features | linear baseline | 1.7538 |  | `python -m src.pipelines.train_text --cv-only --feature-set parser` |
| 2026-04-18 | Codex | TF-IDF session docs on `event_normalized` | ridge text model | 2.6234 |  | `--feature-set tfidf --text-source event_normalized --aggregation session --min-df 2 --max-features 256 --ngram-max 2 --ridge-alpha 10` |
| 2026-04-18 | Codex | TF-IDF session docs on `headline_normalized` | ridge text model | 2.3085 |  | Full headline text underperformed stripped event text. |
| 2026-04-18 | Codex | TF-IDF company-top2 docs on `event_normalized` | ridge text model | 2.0241 |  | Company-aware text docs alone did not beat simple session-level event TF-IDF. |
