IMAGE_NAME = driver-monitoring-system
MODEL_DIR = models
MODEL_FILE = shape_predictor_68_face_landmarks.dat

.PHONY: help build up down restart clean download-model test lint

help:
	@echo "Available commands:"
	@echo "  make build           - Build Docker images"
	@echo "  make up              - Start all services (API, UI, MLflow)"
	@echo "  make down            - Stop all services"
	@echo "  make restart         - Restart containers"
	@echo "  make download-model  - Download pretrained dlib model"
	@echo "  make test            - Run tests with coverage"
	@echo "  make lint            - Check code style"
	@echo "  make clean           - Remove cache and logs"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo " System started!"
	@echo " Dashboard: http://localhost:8501"
	@echo " API Docs:  http://localhost:8000/docs"
	@echo " MLflow:    http://localhost:5000"

down:
	docker-compose down

restart: down up

download-model:
	mkdir -p $(MODEL_DIR)
	@if [ ! -f $(MODEL_DIR)/$(MODEL_FILE) ]; then \
		echo " Downloading dlib model..."; \
		wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O $(MODEL_DIR)/model.dat.bz2; \
		bzip2 -d $(MODEL_DIR)/model.dat.bz2; \
		mv $(MODEL_DIR)/model.dat $(MODEL_DIR)/$(MODEL_FILE); \
		echo " Model ready."; \
	else \
		echo " Model already exists."; \
	fi

test:
	pip install -e ".[dev]" -q
	pytest --cov=app --cov-report=term-missing

lint:
	pip install ruff -q
	ruff check app/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf mlruns/ .coverage htmlcov/ .pytest_cache/