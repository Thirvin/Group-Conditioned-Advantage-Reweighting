SHELL := /bin/bash

UV := uv run python
ROLL_OUT := rollouts_data.pkl
CLUSTERED := clustered_data.pkl

.PHONY: help sync smoke-local full-local smoke-api full-api cluster eval clean

help:
	@echo "Targets:"
	@echo "  make sync        - install/update environment with uv"
	@echo "  make smoke-local - local vLLM smoke test pipeline"
	@echo "  make full-local  - local vLLM full pipeline"
	@echo "  make smoke-api   - Gemini API smoke test pipeline"
	@echo "  make full-api    - Gemini API full pipeline"
	@echo "  make cluster     - run clustering on $(ROLL_OUT)"
	@echo "  make eval        - run metrics on $(CLUSTERED)"
	@echo "  make clean       - remove generated pickle files"

sync:
	uv sync

smoke-local:
	$(UV) 01_generate_rollouts.py --smoke-test
	$(UV) 02_clustering.py --input-path $(ROLL_OUT) --output-path $(CLUSTERED)
	$(UV) 03_metrics_eval.py --input-path $(CLUSTERED)

full-local:
	$(UV) 01_generate_rollouts.py
	$(UV) 02_clustering.py --input-path $(ROLL_OUT) --output-path $(CLUSTERED)
	$(UV) 03_metrics_eval.py --input-path $(CLUSTERED)

smoke-api:
	$(UV) 01_generate_rollouts_api.py --smoke-test
	$(UV) 02_clustering.py --input-path $(ROLL_OUT) --output-path $(CLUSTERED)
	$(UV) 03_metrics_eval.py --input-path $(CLUSTERED)

full-api:
	$(UV) 01_generate_rollouts_api.py
	$(UV) 02_clustering.py --input-path $(ROLL_OUT) --output-path $(CLUSTERED)
	$(UV) 03_metrics_eval.py --input-path $(CLUSTERED)

cluster:
	$(UV) 02_clustering.py --input-path $(ROLL_OUT) --output-path $(CLUSTERED)

eval:
	$(UV) 03_metrics_eval.py --input-path $(CLUSTERED)

clean:
	rm -f $(ROLL_OUT) $(CLUSTERED)
