.PHONY: help install test train serve deploy monitor clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make train        - Run training pipeline"
	@echo "  make serve        - Start local API"
	@echo "  make deploy       - Deploy to Kubernetes"
	@echo "  make monitor      - Start monitoring stack"
	@echo "  make clean        - Clean cache files"

install:
	pip install -r requirements.txt

test:
	pytest tests/unit/ -v --cov=src --cov-report=html
	flake8 src tests --max-line-length=120 || true

serve-local:
	docker-compose -f deployments/docker/docker-compose.yaml up -d

monitor:
	docker-compose -f deployments/docker/docker-compose.yaml up -d prometheus grafana

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov
