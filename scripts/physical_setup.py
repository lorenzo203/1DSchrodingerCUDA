import json
import numpy as np
import sys
import os
from scipy.special import eval_hermite

def main():
    if len(sys.argv) < 2:
        print("Error. Correct Usage: `python physical_setup.py <file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    
    # Load params
    with open(config_file, 'r') as f:
        config = json.load(f)

    N = config["N"]
    L = config["L"]
    x0 = config["x0"]
    sigma = config.get("sigma", 1.0)
    p0 = config["p0"]
    
    dx = L / N
    x = np.linspace(-L/2, L/2 - dx, N) # Create the grid

    # Potential V(x)
    V = np.zeros(N)
    if config["potential_type"] == "barrier":
        start = config["barrier_start"]
        end = config["barrier_end"]
        height = config["barrier_height"]
        V[(x >= start) & (x <= end)] = height
    elif config["potential_type"] == "oscillator":
        omega = config["omega"]
        m = config["mass"]
        V = 0.5 * m * omega**2 * x**2
    elif config["potential_type"] == "well":
        start = config["well_start"]
        end = config["well_end"]
        depth = config["well_depth"]
        V = np.full(N, depth)
        V[(x >= start) & (x <= end)] = 0.0

    # Initial condition for position wavefunction \psi(x, 0)
    if config["potential_type"] == "oscillator" and "n_level" in config:
        n = config["n_level"]
        omega = config["omega"]
        m = config["mass"]
        
        # Adimensional variable in the oscillator
        alpha = np.sqrt(m * omega)
        xi = alpha * (x - x0)
        
        # Hermite Polynomial * Gaussian Exponential
        H_n = eval_hermite(n, xi)
        exp_part = np.exp(-0.5 * xi**2)
        
        # Initial momentum p0
        psi_complex = H_n * exp_part * np.exp(1j * p0 * x)
    else:
        # Fallback for free particle/barrier
        sigma = config.get("sigma", 1.0)
        norm_factor = 1.0 / (2.0 * np.pi * sigma**2)**0.25
        re_exp = - (x - x0)**2 / (4.0 * sigma**2)
        im_exp = p0 * x
        psi_complex = norm_factor * np.exp(re_exp) * np.exp(1j * im_exp)
        
    # Normalization
    prob_totale = np.sum(np.abs(psi_complex)**2) * dx
    psi_complex = psi_complex / np.sqrt(prob_totale)

    # Save data to process it in C++
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    # Scalar parameters for C++
    with open("data/input/params.dat", "w") as f:
        f.write(f"{N} {L} {config['dt']} {config['mass']} {config['num_steps']}\n")

    # Potential (1 column)
    np.savetxt("data/input/potential.dat", V, fmt="%.8f")

    # psi (2 columns: Real and Imaginary part)
    psi_out = np.column_stack((psi_complex.real, psi_complex.imag))
    np.savetxt("data/input/psi_0.dat", psi_out, fmt="%.8f %.8f")

    print(f"Successfully completed setup from {config_file}!")

if __name__ == "__main__":
    main()