# Convenience targets. Every one is a thin wrapper over scripts/, so you can always
# see and rerun the underlying command.

PY      ?= python
PART    ?= eval
STATIC  ?= static
DYNAMIC ?= dynamic

.PHONY: help test install install-frontend data embed cluster train-static train-dynamic \
        train-both score tables figures seeds all clean-derived

help:
	@echo "make test              synthetic end-to-end test (no corpus, no NeMo)"
	@echo "make install           core dependencies"
	@echo "make install-frontend  add NeMo (needed only for 'make embed')"
	@echo ""
	@echo "make data              build the manifest from raw/"
	@echo "make embed             extract 5-scale embeddings   (1-2 GPU-h, ONCE, resumable)"
	@echo "make cluster           NME-SC, prototypes, targets"
	@echo "make train-both        train both ablation arms"
	@echo "make score             decode + zero-collar DER for both arms"
	@echo "make tables            Table 6/7/8 + the statistical table"
	@echo "make figures           fig10-13"
	@echo "make all               everything after 'make data'"
	@echo ""
	@echo "make seeds             3 extra dynamic seeds + variance (paper's open item)"
	@echo "make clean-derived     delete the cache -- forces full re-extraction"

test:
	$(PY) tests/test_pipeline_synthetic.py

install:
	$(PY) -m pip install -r requirements.txt

install-frontend:
	$(PY) -m pip install "nemo_toolkit[asr]==2.0.0"

data:
	$(PY) scripts/01_prepare_data.py

embed:
	$(PY) scripts/02_extract_embeddings.py --dry-run --limit 3
	$(PY) scripts/02_extract_embeddings.py --partition dev
	$(PY) scripts/02_extract_embeddings.py --partition eval

cluster:
	$(PY) scripts/03_cluster_and_targets.py

train-static:
	$(PY) scripts/04_train.py --mode static

train-dynamic:
	$(PY) scripts/04_train.py --mode dynamic

train-both: train-static train-dynamic

score:
	$(PY) scripts/05_infer_and_score.py --run $(STATIC)  --partition $(PART)
	$(PY) scripts/05_infer_and_score.py --run $(DYNAMIC) --partition $(PART)

tables:
	-$(PY) scripts/07_estimate_snr.py --partition $(PART)
	$(PY) scripts/06_ablation_and_stats.py --static $(STATIC) --dynamic $(DYNAMIC)
	$(PY) scripts/08_parameter_budget.py

figures:
	$(PY) scripts/09_figures.py --static $(STATIC) --dynamic $(DYNAMIC)

all: cluster train-both score tables figures

seeds:
	for s in 1235 1236 1237; do \
	  $(PY) scripts/04_train.py --mode dynamic --seed $$s --tag dynamic_s$$s; \
	  $(PY) scripts/05_infer_and_score.py --run dynamic_s$$s --partition $(PART); \
	done
	$(PY) scripts/06_ablation_and_stats.py \
	  --seeds $(DYNAMIC),dynamic_s1235,dynamic_s1236,dynamic_s1237

# Deliberately does NOT touch raw/ -- re-downloading a gated corpus is the one thing
# this project exists to avoid.
clean-derived:
	@echo "This deletes the embedding cache. Re-extraction costs 1-2 GPU-hours."
	@read -p "Type yes to continue: " a; [ "$$a" = yes ] || exit 1
	$(PY) -c "import sys; sys.path.insert(0,'src'); \
	from amsd.platform import resolve_paths; import shutil; \
	p = resolve_paths(); shutil.rmtree(p.derived, ignore_errors=True); \
	print('removed', p.derived)"
