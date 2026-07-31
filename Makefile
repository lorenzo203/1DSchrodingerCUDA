# Quantum Simulator Makefile

# Compiler and flags
NVCC = podman run --rm --userns=keep-id -v "$(CURDIR)":/app:Z -w /app nvidia/cuda:12.3.2-devel-ubuntu22.04 nvcc
CFLAGS = -lcufft

# Paths
TARGET = src/schrodinger
SRC = src/schrodinger.cu
SETUP_SCRIPT = scripts/physical_setup.py
ANIMATE_SCRIPT = scripts/animate.py

# Default experiment
EXP ?= free_particle

$(TARGET): $(SRC)
	@echo "Compiling CUDA source..."
	$(NVCC) $(SRC) -o $(TARGET) $(CFLAGS)

build: $(TARGET)
	@echo "Configuring experiment: $(EXP).json..."
	python $(SETUP_SCRIPT) experiments/$(EXP).json
	@echo "Running simulation..."
	./$(TARGET)

animate:
	@echo "Generating animation..."
	python $(ANIMATE_SCRIPT)

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)
	rm -f data/input/*.dat
	rm -f data/output/*.dat
	rm -f *.gif

.PHONY: build animate clean