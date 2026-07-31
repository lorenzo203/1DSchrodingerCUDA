import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Params
step_max = 1000
step_interval = 100
frames = range(0, step_max, step_interval)

# Figure setup
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-20, 20)      # Space domain of the schrodinger simulation
ax.set_ylim(0, 2.0)       # Relative to max height of the packet from pdf gained in the c++ simulation
ax.set_xlabel("Position (x)")
ax.set_ylabel("Probability Density Function |ψ|²")
ax.set_title("Quantum Wavepacket Evolution")
ax.grid(True, linestyle='--', alpha=0.6)

# Empty line initialization
linea, = ax.plot([], [], lw=2, color='blue', label="Free Particle")
ax.legend()

# Function updating the animation at each frame
def update(step):
    filename = f"output_{step}.dat"
    if os.path.exists(filename):
        # Data: column 0 = x, column 1 = prob
        data = np.loadtxt(filename)
        linea.set_data(data[:, 0], data[:, 1])
        ax.plot(data[:, 0], data[:, 2], color='gray', alpha=0.5)
        ax.set_title(f"Evolution - Step: {step}")
    return linea,

# Animation
anim = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

# Save as GIF
print("GIF generation...")
anim.save('wavepacket.gif', writer='pillow', fps=10)
print("Saved as wavepacket.gif!")

# Screen plot
plt.show()