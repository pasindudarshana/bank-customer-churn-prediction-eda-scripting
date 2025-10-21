.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all help

# Default Python interpreter
PYTHON = python
VENV = .venv\Scripts\Activate.ps1
MLFLOW_PORT ?= 5001

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run the streaming inference pipeline with the sample JSON"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make clean               - Clean up artifacts"

# Install project dependencies and set up environment
install:
	@echo "Installing project dependencies and setting up environment..."
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	@powershell -Command "& { . $(VENV); python -m pip install --upgrade pip }"
	@powershell -Command "& { . $(VENV); pip install -r requirements.txt }"
	@echo "Installation completed successfully!"
	@echo "To activate the virtual environment, run: .\.venv\Scripts\Activate.ps1"

# Clean up
clean:
	@echo "Cleaning up artifacts..."
	rm -rf artifacts/models/*
	rm -rf artifacts/evaluation/*
	rm -rf artifacts/predictions/*
	rm -rf artifacts/encode/*
	rm -rf mlruns
	@echo "Cleanup completed!"

# Run data pipeline
data-pipeline:
	@echo "Start running data pipeline..."
	@powershell -Command "& { . $(VENV); $(PYTHON) pipelines/data_pipeline.py }"
	@echo "Data pipeline completed successfully!"

.PHONY: data-pipeline-rebuild
data-pipeline-rebuild:
	@powershell -Command "& { . $(VENV); $(PYTHON) -c \"from pipelines.data_pipeline import data_pipeline; data_pipeline(force_rebuild=True)\" }"

# Run training pipeline
train-pipeline:
	@echo "Running training pipeline..."
	@powershell -Command "& { . $(VENV); $(PYTHON) pipelines/training_pipeline.py }"

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo "Running streaming inference pipeline with sample JSON..."
	@powershell -Command "& { . $(VENV); $(PYTHON) pipelines/streaming_inference_pipeline.py }"

# Run all pipelines in sequence
run-all:
	@echo "Running all pipelines in sequence..."
	@echo "========================================"
	@echo "Step 1: Running data pipeline"
	@echo "========================================"
	@powershell -Command "& { . $(VENV); python pipelines/data_pipeline.py }"
	@echo "\n========================================"
	@echo "Step 2: Running training pipeline"
	@echo "========================================"
	@powershell -Command "& { . $(VENV); python pipelines/training_pipeline.py }"
	@echo "\n========================================"
	@echo "Step 3: Running streaming inference pipeline"
	@echo "========================================"
	@powershell -Command "& { . $(VENV); python pipelines/streaming_inference_pipeline.py }"
	@echo "\n========================================"
	@echo "All pipelines completed successfully!"
	@echo "========================================"

mlflow-ui:
	@echo "Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the server"
	@powershell -Command "& { . $(VENV); mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT) }"

# Stop all running MLflow servers
stop-all:
	@echo "Stopping all MLflow servers..."
	$mlflow_procs = Get-NetTCPConnection -LocalPort $(MLFLOW_PORT) -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique; if ($mlflow_procs) { Stop-Process -Id $mlflow_procs -Force; Write-Host '✓ Killed MLflow processes on port $(MLFLOW_PORT)'; } else { Write-Host 'No processes found on port $(MLFLOW_PORT)'; }
	$procs1 = Get-Process -Name '*mlflow*' -ErrorAction SilentlyContinue; if ($procs1) { Stop-Process -InputObject $procs1 -Force }
	$procs2 = Get-Process -Name '*gunicorn*' -ErrorAction SilentlyContinue; if ($procs2) { Stop-Process -InputObject $procs2 -Force }
	@echo "✅ All MLflow servers have been stopped"