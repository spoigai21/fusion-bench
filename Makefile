# fusion-bench — the whole pipeline, in the order the phases run.
#
#   make sim        CPU logic simulation — no GPU needed  (run this before renting)
#   make phase0     GPU + Nsight counter access        (do this first, on the rented box)
#   make build      compile the extension              (phase 1 checkpoint)
#   make verify     FP64 cross-check, all edge cases   (phase 2)
#   make bench      timings   -> results/summary.csv   (phase 6)
#   make profile    counters  -> results/raw/ncu_*.csv (phase 6)
#   make bytes      byte table -> results/bytes.csv
#   make plot       chart     -> results/plots/
#   make all        verify + bench + profile + bytes + plot
#   make analytic   Plan B byte table, when counters are blocked

PYTHON ?= python
comma := ,
SHAPES ?= 4096x1024,4096x8192

.PHONY: sim phase0 build verify bench profile bytes analytic plot all quick clean distclean

sim:
	$(PYTHON) tests/sim_logic.py

phase0:
	bash scripts/phase0_check.sh

build:
	$(PYTHON) -m bench.ext

verify:
	$(PYTHON) tests/verify.py

bench:
	$(PYTHON) bench/run_all.py --shapes $(SHAPES)

quick:
	$(PYTHON) bench/run_all.py --shapes $(SHAPES) --warmup 5 --iters 20 --skip-compile

profile:
	bash scripts/profile.sh $(subst $(comma), ,$(SHAPES))

bytes:
	$(PYTHON) bench/parse_ncu.py

analytic:
	$(PYTHON) bench/parse_ncu.py --analytic --shapes $(SHAPES)

plot:
	$(PYTHON) bench/plot.py

all: verify bench profile bytes plot

clean:
	rm -rf .build __pycache__ bench/__pycache__ tests/__pycache__

distclean: clean
	rm -f results/summary.csv results/bytes.csv results/env.json
	rm -f results/raw/*.csv results/plots/*.png
