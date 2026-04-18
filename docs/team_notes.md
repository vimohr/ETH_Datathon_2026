# Team Notes

## Ownership

- Infrastructure: `src/data/`, `src/submission.py`, `Makefile`, repo hygiene, output paths.
- Pricing: `src/features/price.py`, price EDA, session-level price heuristics, feature expansion.
- Modeling and uncertainty: `src/models/`, `src/evaluation/`, position sizing, calibration, CV interpretation.
- Headlines: `src/features/headlines.py`, headline parsing, relevance filtering, text-driven features.

## Immediate Priorities

- Lock a shared validation protocol before comparing models.
- Keep all generated submissions under `outputs/submissions/`.
- Save OOF predictions under `outputs/oof/` so blends can be built quickly.
- Treat the root-level scripts and notebook as legacy references, not the primary workflow.
- Text embedding and transformer experiments live on `exp/text-embeddings` in `/scratch/tzerweck/datathon/ETH_Datathon_2026_embeddings`.
- The concrete NLP experiment ladder is documented in `docs/text_embeddings_plan.md`.
