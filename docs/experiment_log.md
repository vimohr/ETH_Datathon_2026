# Experiment Log

| Timestamp | Owner | Features | Model | CV Sharpe | Public LB | Notes |
|-----------|-------|----------|-------|-----------|-----------|-------|
| 2026-04-18 | Codex | parser baseline headline features | linear baseline | 1.7538 |  | `python -m src.pipelines.train_text --cv-only --feature-set parser` |
| 2026-04-18 | Codex | TF-IDF session docs on `event_normalized` | ridge text model | 2.6234 |  | `--feature-set tfidf --text-source event_normalized --aggregation session --min-df 2 --max-features 256 --ngram-max 2 --ridge-alpha 10` |
| 2026-04-18 | Codex | TF-IDF session docs on `headline_normalized` | ridge text model | 2.3085 |  | Full headline text underperformed stripped event text. |
| 2026-04-18 | Codex | TF-IDF company-top2 docs on `event_normalized` | ridge text model | 2.0241 |  | Company-aware text docs alone did not beat simple session-level event TF-IDF. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, session pooling on `event_normalized` | ridge text model | 2.7422 |  | Best text-only result so far. Cached by unique normalized event text. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, company-top2 pooling on `event_normalized` | ridge text model | 1.8436 |  | Company-aware pooling underperformed simple session pooling with current relevance ranking. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, company-top2 pooling + structured relevance features | ridge text model | 1.1608 |  | Adding current structured company features hurt further. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, `company_weighted` pooling on `event_normalized` | ridge text model | 2.7745 |  | Best text-only result so far. Weighted company pooling beats simple session pooling. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, `company_weighted` pooling + price-reaction-aware relevance | ridge text model | 2.7309 |  | Seen-bar price reactions did not help the weighted pool. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, `company_top2` pooling + price-reaction-aware relevance | ridge text model | 2.5309 |  | Price reactions materially improved `company_top2`, but still underperformed `company_weighted`. |
| 2026-04-18 | Codex | Price baseline + `company_weighted` MiniLM blend | OOF position blend | 2.8558 |  | Best search: weights `{price: 0.3, textcw: 0.7}`, neutral band `0`, dispersion gamma `0`. |
| 2026-04-18 | Codex | Price baseline + `company_weighted` MiniLM + session MiniLM blend | OOF position blend | 2.8700 |  | Best search: weights `{price: 0.2, textcw: 0.5, textsess: 0.3}`, neutral band `15`, dispersion gamma `0`. |
