# Quantum Simulator Makefile

# Compiler and flags
NVCC = podman run --rm --userns=keep-id -v "$(CURDIR)":/app:Z -w /app nvidia/cuda:12.3.2-devel-ubuntu22.04 nvcc
CFLAGS = -lcufft

# Paths
TARGET = src/schrodinger
SRC = src/schrodinger.cu
SETUP_SCRIPT = scripts/physical_setup.py
ANIMATE_SCRIPT = scripts/animate2.py

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Create venv and install dependencies
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install numpy matplotlib physics-plot scipy
	@echo "Setup completed. Virtual Environment ready."

# Default experiment
EXP ?= quantum_well

$(TARGET): $(SRC)
	@echo "Compiling CUDA source..."
	$(NVCC) $(SRC) -o $(TARGET) $(CFLAGS)

build: $(TARGET)
	@echo "Configuring experiment: $(EXP).json..."
	$(PYTHON) $(SETUP_SCRIPT) experiments/$(EXP).json
	@echo "Running simulation..."
	./$(TARGET)

animate:
	@echo "Generating animation..."
	$(PYTHON) $(ANIMATE_SCRIPT)

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)
	rm -f data/input/*.dat
	rm -f data/output/*.dat
	rm -f *.gif

.PHONY: build animate clean