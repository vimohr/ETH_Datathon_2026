PYTHON ?= .venv/bin/python
BASELINE_PUBLIC ?= outputs/submissions/baseline_public.csv
BASELINE_PRIVATE ?= outputs/submissions/baseline_private.csv

.PHONY: install dirs cv-baseline baseline-public baseline-private

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
