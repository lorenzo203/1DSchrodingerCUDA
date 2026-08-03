#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuComplex.h>
#include <cufft.h>
#include <fstream>

using namespace std;

// Potential kernel (position representation)
__global__ void apply_potential(cuDoubleComplex* d_psi, const double* d_V, double dt, int N) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Assert we are in array's boundaries
    if (idx < N) {
        double V_val = d_V[idx];
        
        // Half time step
        double phase = - V_val * dt /2;
        
        // e^(i*phase) = cos(phase) + i*sin(phase)
        cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase), sin(phase));
        
        // Multiply initial wavefunction psi for the phase factor
        d_psi[idx] = cuCmul(d_psi[idx], phase_factor);
    }
}

// Kernel for kinetic energy in momentum representation
__global__ void apply_kinetic(cuDoubleComplex* d_psi, const double* d_p, double dt, double mass, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        double p_val = d_p[idx];
        
        double K = p_val * p_val / (2 * mass);
        
        double phase = -K * dt;
        
        cuDoubleComplex phase_factor = make_cuDoubleComplex(cos(phase)/N,sin(phase)/N);
        
        d_psi[idx] = cuCmul(d_psi[idx], phase_factor);
    }
}

int main() {
    // Global params reading
    ifstream file_params("data/input/params.dat");
    if (!file_params) {
        cerr << "Error: params.dat unavailable. Check Python execution." << endl;
        return 1;
    }
    
    int N, num_steps;
    double L, dt, m;
    file_params >> N >> L >> dt >> m >> num_steps;
    file_params.close();

    const double dx = L / N;            
    const double dp = 2 * M_PI / L;   

    vector<complex<double>> h_psi(N); 
    vector<double> h_V(N);             
    vector<double> h_p(N); 

    // Potential
    ifstream file_V("data/input/potential.dat");
    if (!file_V) { cerr << "Error in potential.dat opening" << endl; return 1; }
    for (int j = 0; j < N; j++) {
        file_V >> h_V[j];
    }
    file_V.close();

    // Initial Wavefunction
    ifstream file_psi("data/input/psi_0.dat");
    if (!file_psi) { cerr << "Error opening psi_0.dat" << endl; return 1; }
    for (int j = 0; j < N; j++) {
        double real_part, imag_part;
        file_psi >> real_part >> imag_part;
        h_psi[j] = complex<double>(real_part, imag_part);
    }
    file_psi.close();

    // Momentum array
    for (int n = 0; n < N; n++) {
        if (n < N / 2) {
            h_p[n] = n * dp;
        } else {
            h_p[n] = (n - N) * dp;
        }
    }

    cout << "Setup loaded through Python. Params: N=" << N << " L=" << L << endl;

    // Sorting for FFT in momentum space
    for (int n = 0; n < N; n++) {
        if (n < N / 2) {
            h_p[n] = n * dp;
        } else {
            h_p[n] = (n - N) * dp;
        }
    }

    cout << "Setup complete, ready for FFT execution through the GPU." << endl;


    cuDoubleComplex* d_psi;
    double *d_V, *d_p;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_psi, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_V, N * sizeof(double));
    cudaMalloc((void**)&d_p, N * sizeof(double));

    // Copy data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_psi, h_psi.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // cuFFT setup
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1); // Z2Z means double-precision Complex-to-Complex

    // Kernel launch
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    cout << "Starting time evolution..." << endl;

    // main loop
    for (int step = 0; step < num_steps; step++) {
        
        // Half step of potential in position space
        apply_potential<<<num_blocks, threads_per_block>>>(d_psi, d_V, dt, N);

        // FFT from position to momentum space
        // in-place=> idata and odata are both d_psi. Direction is CUFFT_FORWARD.
        cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_FORWARD);

        // Full step of kinetic energy in momentum space
        apply_kinetic<<<num_blocks, threads_per_block>>>(d_psi, d_p, dt, m, N);

        // IFFT from momentum to position space
        cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE);

        // Half step of potential in position space
        apply_potential<<<num_blocks, threads_per_block>>>(d_psi, d_V, dt, N);
        

        // Every 20 steps copy d_psi back to h_psi and save it to a file
        if (step % 20 == 0) {
            // Move updated array from GPU to CPU
            cudaMemcpy(h_psi.data(), d_psi, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
            
            // File preparation (eg: output_0.dat, output_100.dat)
            string filename = "data/output/output_" + to_string(step) + ".dat";
            ofstream file(filename);
            
            double total_prob = 0.0;
            
            // Compute and write probability
            for (int j = 0; j < N; j++) {
                double x = -L / 2.0 + j * dx;
                double prob_density = norm(h_psi[j]); 
                
                // Rectangular integration
                total_prob += prob_density * dx; 
                
                /* Format for GNUPlot/Matplotlib: "x probability potential" separated by a space
                   The value for potential is divided by 10 or more in order to ease the visual representation */
                file << x << " " << prob_density << " " << h_V[j]/10 << "\n";
            }
            file.close();
           
            // Terminal check to ensure total prob of each pdf (integral over the domain) is 1
            cout << "Step: " << step << " | Norm: " << total_prob << endl;
        }
    }

    // cleanup
    cufftDestroy(plan);
    cudaFree(d_psi);
    cudaFree(d_V);
    cudaFree(d_p);

    cout << "Simulation finished." << endl;
    return 0;
}