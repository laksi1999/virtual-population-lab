PYTHON	?= python3.13
PIP		?= pip3.13
CONFIG	?= apple_quality

venv:
	$(PYTHON) -m venv venv
	venv/bin/$(PYTHON) -m pip install --upgrade pip setuptools

.PHONY: setup
setup: venv
	venv/bin/$(PIP) install -r requirements.txt

.PHONY: run
run:
	venv/bin/$(PYTHON) -m src.main $(CONFIG)

.PHONY: check-overfit
check-overfit:
	venv/bin/$(PYTHON) -m src.check_generalization $(CONFIG)
