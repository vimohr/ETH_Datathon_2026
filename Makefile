PYTHON ?= .venv/bin/python
BASELINE_PUBLIC ?= outputs/submissions/baseline_public.csv
BASELINE_PRIVATE ?= outputs/submissions/baseline_private.csv
PUBLIC_SPLIT_FILE ?= outputs/submissions/latest_public_test.csv
PRIVATE_SPLIT_FILE ?= outputs/submissions/latest_private_test.csv
COMPETITION_FILE ?= outputs/submissions/latest_competition.csv
COMPETITION ?= hrt-eth-zurich-datathon-2026
SUBMISSION_MESSAGE ?= manual submission

.PHONY: install dirs cv-baseline baseline-public baseline-private baseline-competition combine-submission kaggle-submit kaggle-status

install:
	$(PYTHON) -m pip install -r requirements.txt

dirs:
	mkdir -p outputs/models outputs/oof outputs/figures outputs/submissions notebooks/99_scratch

cv-baseline:
	$(PYTHON) -m src.pipelines.train_baseline --cv-only

baseline-public:
	$(PYTHON) -m src.pipelines.train_baseline --test-split public_test --output $(BASELINE_PUBLIC)

baseline-private:
	$(PYTHON) -m src.pipelines.train_baseline --test-split private_test --output $(BASELINE_PRIVATE)

baseline-competition: baseline-public baseline-private combine-submission

combine-submission:
	$(PYTHON) -m src.pipelines.competition combine --public $(PUBLIC_SPLIT_FILE) --private $(PRIVATE_SPLIT_FILE) --output $(COMPETITION_FILE)

kaggle-submit:
	$(PYTHON) -m src.pipelines.competition submit --file $(COMPETITION_FILE) --competition $(COMPETITION) --message "$(SUBMISSION_MESSAGE)"

kaggle-status:
	$(PYTHON) -m src.pipelines.competition status --competition $(COMPETITION)
