PYTHON ?= .venv/bin/python
BASELINE_PUBLIC ?= outputs/submissions/baseline_public.csv
BASELINE_PRIVATE ?= outputs/submissions/baseline_private.csv
TEXT_PUBLIC ?= outputs/submissions/text_public.csv
TEXT_PRIVATE ?= outputs/submissions/text_private.csv

.PHONY: install dirs cv-baseline baseline-public baseline-private cv-text text-public text-private

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
