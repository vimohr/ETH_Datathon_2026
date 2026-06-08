# Experiment Log

| Timestamp | Owner | Features | Model | CV Sharpe | Public LB | Notes |
|-----------|-------|----------|-------|-----------|-----------|-------|
| 2026-04-18 | Codex | TF-IDF text features, session aggregation, event-normalized preprocessing | tfidf_session_event_normalized | 2.6234 |  | Result from `ETH_Datathon_2026_embeddings`. Strongest non-embedding text-only baseline in that repo. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, session aggregation, event-normalized preprocessing | minilm_session_event_normalized | 2.7422 |  | Result from `ETH_Datathon_2026_embeddings`. Best text-only CV result so far. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, company-top2 aggregation | minilm_company_top2 | 1.8436 |  | Result from `ETH_Datathon_2026_embeddings`. Worse than session-level embedding aggregation. |
| 2026-04-18 | Codex | Frozen MiniLM embeddings, company-top2 aggregation plus structured relevance features | minilm_company_top2_relevance | 1.1608 |  | Result from `ETH_Datathon_2026_embeddings`. Relevance feature stack underperformed badly in this configuration. |
| 2026-04-18 | Codex | Headline parser + company relevance weighting + trimmed categorical text features | text_only | 1.7538 | 0.71617 | Final combined Kaggle upload used both test splits (`20000` rows). Standalone text-only signal was real in local CV but not competitive on the public board. |
| 2026-04-18 | Codex | Headline parser + company relevance weighting + trimmed categorical text features | text_only | 1.7538 | ERROR | Initial Kaggle upload failed because only `public_test` was submitted (`10000` rows); Kaggle expected combined `public_test + private_test` (`20000` rows). |
