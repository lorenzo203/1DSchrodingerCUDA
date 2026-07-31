import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Params
step_max = 2000
step_interval = 20
frames = range(0, step_max, step_interval)

# Figure setup
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-20, 20)      # Space domain of the schrodinger simulation
ax.set_ylim(0, 1.0)       # Relative to max height of the packet from pdf gained in the c++ simulation
ax.set_xlabel("Position (x)")
ax.set_ylabel("Probability Density Function |ψ|² / Scaled Potential")
ax.set_title("Quantum Wavepacket Evolution")
ax.grid(True, linestyle='--', alpha=0.6)

# Empty line initialization
line, = ax.plot([], [], lw=2, color='blue', label="Particle")
line_V, = ax.plot([], [], lw=2, color='gray', alpha=0.5, label="Potential")
ax.legend()

# Function updating the animation at each frame
def update(step):
    filename = f"../data/output/output_{step}.dat"
    if os.path.exists(filename):
        # Data: column 0 = x, column 1 = prob, column 2 = potential
        data = np.loadtxt(filename)
        line.set_data(data[:, 0], data[:, 1])
        if data.shape[1] > 2:
            line_V.set_data(data[:, 0], data[:, 2])
        ax.set_title(f"Evolution - Step: {step}")
    return line, line_V

# Animation
anim = animation.FuncAnimation(fig, update, frames=frames, interval=300, blit=True)

# Save as GIF
print("GIF generation...")
anim.save('wavepacket.gif', writer='pillow', fps=10)
print("Saved as wavepacket.gif!")

# Screen plot
plt.show()