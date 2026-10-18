.PHONY: test demo run
test:
	python -m pytest -q
demo:
	PYTHONPATH=src python examples/demo.py
run:
	uvicorn mlplatform.api:app --host 0.0.0.0 --port 8080
