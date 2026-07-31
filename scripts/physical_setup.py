import json
import numpy as np
import sys
import os

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
    sigma = config["sigma"]
    p0 = config["p0"]
    
    dx = L / N
    x = np.linspace(-L/2, L/2 - dx, N) # Create the grid

    # Potential V(x)
    V = np.zeros(N)
    if config["potential_type"] == "barrier":
        start = config["barrier_start"]
        end = config["barrier_end"]
        height = config["barrier_height"]
        # Apply it only for points inside the barrier
        V[(x >= start) & (x <= end)] = height

    # Initial condition for position wavefunction \psi(x, 0)
    norm_factor = 1.0 / (2.0 * np.pi * sigma**2)**0.25
    re_exp = - (x - x0)**2 / (4.0 * sigma**2)
    im_exp = p0 * x
    
    psi_complex = norm_factor * np.exp(re_exp) * np.exp(1j * im_exp)
    
    # Normalization
    prob_totale = np.sum(np.abs(psi_complex)**2) * dx
    psi_complex = psi_complex / np.sqrt(prob_totale)

    # Save data to process it in C++
    os.makedirs("../data/input", exist_ok=True)
    os.makedirs("../data/output", exist_ok=True)

    # Scalar parameters for C++
    with open("../data/input/params.dat", "w") as f:
        f.write(f"{N} {L} {config['dt']} {config['mass']} {config['num_steps']}\n")

    # Potential (1 column)
    np.savetxt("../data/input/potential.dat", V, fmt="%.8f")

    # psi (2 columns: Real and Imaginary part)
    psi_out = np.column_stack((psi_complex.real, psi_complex.imag))
    np.savetxt("../data/input/psi_0.dat", psi_out, fmt="%.8f %.8f")

    print(f"Successfully completed setup from {config_file}!")

if __name__ == "__main__":
    main()