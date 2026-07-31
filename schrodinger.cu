#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuComplex.h>
#include <cufft.h>
#include <fstream>
#include <bits/stdc++.h>

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
    const int N = 2048;                 
    const double L = 40.0;              
    
    // Compute spacial steps (dx) and momentum steps (dp)
    // dp = 2*pi / L
    const double dx = L / N ;            
    const double dp = 2 * M_PI / L ;   

    const double dt = 0.01;
    const int m = 1;
    const int num_steps = 1000;

    /* Here the parameters that define the initial condition
     * for the wavefunction are defined.                        */
    const double x0 = -10.0;            
    const double sigma = 1.0;           
    const double p0 = 5.0; //=>The kinetic energy Ek in this case is 12.5, so we need V>12.5 to see tunnel effect  
    
    /* QUANTUM WALL PARAMETERS */
    const double V0 = 15;
    const double wall_start = 0.0;
    const double wall_end   = 2.0;

    vector<complex<double>> h_psi(N); // vector containing all the wavefunctions in each interval  
    vector<double> h_V(N); //vector containing the potentials             
    vector<double> h_p(N); //vector containing the positions used to map values to the output of FFT           

    double total_prob = 0.0; 

    // Iterate on each point (interval) of the discretized domain
    for (int j = 0; j < N; j++) {
        // Compute x in order to center the interval in 0
        double x = - L / 2 + j * dx;

        // Determine the potential in that point
        if(x >= wall_start && x <= wall_end){
            h_V[j] = V0;
        }else{
            h_V[j] = 0.0;
        }
        
        double norm_factor = 1 / pow(2 * M_PI * sigma * sigma, 0.25);
        double re_exp = - (x - x0) * (x - x0) / (4 * sigma * sigma);
        complex<double> im_exp(0.0 ,  p0 * x); 
        
        // Put together the whole initial condition of the wavefunction
        h_psi[j] = norm_factor * exp(re_exp) * exp(im_exp);

        /* Add the norm of the current wavefunction to the 
         * total_prob variable to take into account of
         * a normalization factor in the end to impose ||\psi(x_j)||=1 for each j*/
        total_prob += norm(h_psi[j]) * dx; 
    }

    // Normalization
    for (int j = 0; j < N; j++) {
        h_psi[j] = h_psi[j] / sqrt(total_prob);
    }

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
        // We do it in-place, so idata and odata are both d_psi. Direction is CUFFT_FORWARD.
        cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_FORWARD);

        // Full step of kinetic energy in momentum space
        apply_kinetic<<<num_blocks, threads_per_block>>>(d_psi, d_p, dt, m, N);

        // IFFT from momentum to position space
        cufftExecZ2Z(plan, d_psi, d_psi, CUFFT_INVERSE);

        // Half step of potential in position space
        apply_potential<<<num_blocks, threads_per_block>>>(d_psi, d_V, dt, N);
        

        // Every 100 steps copy d_psi back to h_psi and save it to a file
        if (step % 100 == 0) {
            // Move updated array from GPU to CPU
            cudaMemcpy(h_psi.data(), d_psi, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
            
            // File preparation (eg: output_0.dat, output_100.dat)
            string filename = "output_" + to_string(step) + ".dat";
            ofstream file(filename);
            
            double total_prob = 0.0;
            
            // Compute and write probability
            for (int j = 0; j < N; j++) {
                double x = -L / 2.0 + j * dx;
                double prob_density = norm(h_psi[j]); 
                
                // Rectangular integration
                total_prob += prob_density * dx; 
                
                /* Format for GNUPlot/Matplotlib: "x probability potential" separated by a space
                   The value for potential is divided by 50 in order to ease the visual representation */
                file << x << " " << prob_density << " " << h_V[j]/25 << "\n";
            }
            file.close();
            sort(h_V.begin(),h_V.end());
            // Terminal check to ensure total prob of each pdf is 1
            cout << "Step: " << step << " | Norm: " << total_prob << " Max value of pdf: " << h_V.back() << endl;
        }
    }

    // cleanup
    cufftDestroy(plan);
    cudaFree(d_psi);
    cudaFree(d_V);
    cudaFree(d_p);

    cout << "Simulation finished!" << endl;
    return 0;
}