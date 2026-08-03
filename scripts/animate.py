import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import physics_plot

# Apply the academic style sheet globally
plt.style.use("physics_plot.pp_base")

step_max = 800
step_interval = 20
frames = range(0, step_max, step_interval)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-50, 50)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r"Position $x$")
ax.set_ylabel(r"Probability Density $|\psi(x,t)|^2$ / Scaled Potential $V(x)$")
ax.set_title("Quantum Wavepacket Evolution")

line, = ax.plot([], [], lw=2, color='blue', label=r"Particle $|\psi|^2$")
line_V, = ax.plot([], [], lw=2, color='gray', alpha=0.5, label=r"Potential $V(x)$")
ax.legend(loc="upper right")

def update(step):
    filename = f"data/output/output_{step}.dat"
    if os.path.exists(filename):
        data = np.loadtxt(filename)
        line.set_data(data[:, 0], data[:, 1])
        if data.shape[1] > 2:
            line_V.set_data(data[:, 0], data[:, 2])
        ax.set_title(f"Evolution - Step: {step}")
    return line, line_V

anim = animation.FuncAnimation(fig, update, frames=frames, interval=300, blit=True)

print("Generating GIF...")
anim.save('wavepacket.gif', writer='pillow', fps=10)
print("Saved as wavepacket.gif!")