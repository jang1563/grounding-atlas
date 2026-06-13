.PHONY: help install data lint fix figures clean

help:
	@echo "make install  - install Python dependencies"
	@echo "make data     - fetch large reference DBs (per-branch setup scripts)"
	@echo "make lint     - ruff check"
	@echo "make fix      - ruff check --fix (safe autofixes)"
	@echo "make figures  - regenerate the synthesis figure"
	@echo "make clean    - remove Python caches"

install:
	pip install -r requirements.txt

data:
	bash variant_grounding/eval/setup_data_cayuga.sh
	bash protein_grounding/eval/setup_data_cayuga.sh

lint:
	ruff check .

fix:
	ruff check --fix .

figures:
	python eval/make_synthesis_figure.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
