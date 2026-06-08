# Team Notes

## Ownership

- Infrastructure: `src/data/`, `src/submission.py`, `Makefile`, repo hygiene, output paths.
- Pricing: `src/features/price.py`, price EDA, session-level price heuristics, feature expansion.
- Modeling and uncertainty: `src/models/`, `src/evaluation/`, position sizing, calibration, CV interpretation.
- Headlines: `src/features/headlines.py`, headline parsing, relevance filtering, text-driven features.

## Immediate Priorities

- Lock a shared validation protocol before comparing models.
- Keep all generated submissions under `outputs/submissions/`.
- Kaggle uploads must use a combined `latest_competition.csv` with both test splits.
- Save OOF predictions under `outputs/oof/` so blends can be built quickly.
- Treat the root-level scripts and notebook as legacy references, not the primary workflow.
