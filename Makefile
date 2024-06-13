sources = ood_robustness
conda_file = environment.yaml
lock_file = environment.lock

plattform = linux-64
ifeq ($(OS),Windows_NT)
	plattform = win-64
endif
ifeq ($(OS),Darwin)
	plattform = osx-64
endif

conda_lock = no
CLF := $(shell command -v conda-lock 2> /dev/null)
ifdef CLF
	conda_lock = yes
endif
ifeq ($(wildcard ./$(lock_file)),)
	conda_lock = no
endif

.PHONY: environment environment_lock test format lint unittest docs coverage pre-commit clean

environment:
ifeq ($(conda_lock), yes)
	conda-lock render -p $(plattform) $(lock_file)
	conda create -n ood-robustness --file conda-$(plattform).lock
	rm -rf conda-$(plattform).lock
else
	conda create -n ood-robustness python=3.11
	conda env update -n ood-robustness --file $(conda_file)
endif
	conda activate ood-robustness
	pre-commit install

environment_lock:
	conda-lock --mamba --lockfile $(lock_file) -f $(conda_file)

test: format lint unittest

format:
	isort $(sources) tests
	black $(sources) tests

lint:
	flake8 $(sources) tests
	mypy $(sources) tests

unittest:
	pytest

docs:
	mkdocs build

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage
	rm -rf conda-$(plattform).lock
