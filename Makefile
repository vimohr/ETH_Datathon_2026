PYTHON ?= $(or $(wildcard .venv/bin/python),$(wildcard ../ETH_Datathon_2026/.venv/bin/python),python3)
BASELINE_PUBLIC ?= outputs/submissions/baseline_public.csv
BASELINE_PRIVATE ?= outputs/submissions/baseline_private.csv
TEXT_PUBLIC ?= outputs/submissions/text_public.csv
TEXT_PRIVATE ?= outputs/submissions/text_private.csv
PUBLIC_SPLIT_FILE ?= outputs/submissions/latest_public_test.csv
PRIVATE_SPLIT_FILE ?= outputs/submissions/latest_private_test.csv
COMPETITION_FILE ?= outputs/submissions/latest_competition.csv
COMPETITION ?= hrt-eth-zurich-datathon-2026
SUBMISSION_MESSAGE ?= manual submission

.PHONY: install dirs cv-baseline baseline-public baseline-private cv-text text-public text-private baseline-competition combine-submission kaggle-submit kaggle-status

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

cv-text:
	$(PYTHON) -m src.pipelines.train_text --cv-only

text-public:
	$(PYTHON) -m src.pipelines.train_text --test-split public_test --output $(TEXT_PUBLIC)

text-private:
	$(PYTHON) -m src.pipelines.train_text --test-split private_test --output $(TEXT_PRIVATE)

baseline-competition: baseline-public baseline-private combine-submission

combine-submission:
	$(PYTHON) -m src.pipelines.competition combine --public $(PUBLIC_SPLIT_FILE) --private $(PRIVATE_SPLIT_FILE) --output $(COMPETITION_FILE)

kaggle-submit:
	$(PYTHON) -m src.pipelines.competition submit --file $(COMPETITION_FILE) --competition $(COMPETITION) --message "$(SUBMISSION_MESSAGE)"

kaggle-status:
	$(PYTHON) -m src.pipelines.competition status --competition $(COMPETITION)
