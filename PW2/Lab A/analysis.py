import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# --- 1. Load Data ---
t, y = np.loadtxt('freefall.csv', delimiter=',', skiprows=1, unpack=True)

# --- 2. Numerical Differentiation ---
v = np.gradient(y, t)
a = np.gradient(v, t)

# --- 3. Statistical Analysis ---
mean_a = np.mean(a)
std_a = np.std(a)
print(f"Mean Acceleration: {mean_a:.2f} m/s^2")
print(f"Acceleration Std Dev: {std_a:.2f} m/s^2")

# --- 4. Numerical Integration Back ---
v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]
max_diff = np.max(np.abs(y - y_rec))
print(f"Max Position Recovery Difference: {max_diff:.2f} m")

# --- 5. Generate 3-Panel Stacked Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Panel 1: Position
ax1.plot(t, y, label='Position $y(t)$', color='blue')
ax1.set_ylabel('Position (m)')
ax1.set_title('Freefall Motion Analysis')
ax1.legend(loc='upper right')
ax1.grid(True)

# Panel 2: Velocity
ax2.plot(t, v, label='Velocity $v(t)$', color='orange')
ax2.set_ylabel('Velocity (m/s)')
ax2.legend(loc='upper right')
ax2.grid(True)

# Panel 3: Acceleration
ax3.plot(t, a, label='Acceleration $a(t)$', color='red')
ax3.axhline(-9.81, color='black', linestyle='--', label='Theoretical $g = -9.81 \\text{ m/s}^2$')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (m/s²)')
ax3.legend(loc='upper right')
ax3.grid(True)

plt.tight_layout()
plt.savefig('motion.png', dpi=300)
print("Saved plot as motion.png")
