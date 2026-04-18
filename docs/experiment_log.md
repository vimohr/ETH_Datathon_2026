# Experiment Log

| Timestamp | Owner | Features | Model | CV Sharpe | Public LB | Notes |
|-----------|-------|----------|-------|-----------|-----------|-------|
| 2026-04-18 | Codex | Headline parser + company relevance weighting + trimmed categorical text features | text_only | 1.7538 | 0.71617 | Final combined Kaggle upload used both test splits (`20000` rows). Standalone text-only signal was real in local CV but not competitive on the public board. |
| 2026-04-18 | Codex | Headline parser + company relevance weighting + trimmed categorical text features | text_only | 1.7538 | ERROR | Initial Kaggle upload failed because only `public_test` was submitted (`10000` rows); Kaggle expected combined `public_test + private_test` (`20000` rows). |
