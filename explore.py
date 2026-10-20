

#Functionality: 
    #Generates a comprehensive, wide-domain global phase portrait for the 2 DOF glucose-insulin control model. It computes analytical equilibrium points, evaluates Jacobian eigenvalues for stability classification, 
    #constructs nullclines and vector field streamlines, and traces trajectories near the saddle point separatrix to map the basin of attraction and safety boundaries.

#Inputs:
    #Relies on the external module 'g_simul' (imported as gs) for system parameter count (gs.nPar), Jacobian matrix evaluation (gs.Jac), and 2nd-order 
    #Runge-Kutta numerical integration steps (gs.RK2step).
    #Internal parameters: step size (h = 0.001), regulatory coefficients (p[0] to p[2]), and external glucose input rate (p[3] = G).

#Outputs:
    #Saves a high-resolution publication-grade image file: 'global_phase_portrait_wide.png' (300 DPI).
    #Renders an interactive Matplotlib figure window displaying the phase portrait.

#Usage:
    #Run the script directly in a Python environment where 'g_simul' is available

# parameters
# stepsize
h = 0.001
# storage array for parameteres
p = np.zeros((gs.nPar,)) 
p[0] = 0.1  # a, glucose-insulin regulation rate
p[1] = 1.0  # b, insulin effectiveness scale
p[2] = 1.0  # c, decay/clearance parameter
p[3] = 2.0  # G, external glucose input rate

# Calc discriminant
D = (p[0] + p[2])**2 + 4.0 * p[0] * p[2] * p[3]
# coordinates of upper equilibiurm points
eq0 = [
    -(p[0] + p[2])/(2.0 * p[0]) + np.sqrt(D)/(2.0 * p[0]),
    -(p[0] + p[2])/(2.0 * p[2]) + np.sqrt(D)/(2.0 * p[2])
]
# coordinates of lower equilibiurm points
eq1 = [
    -(p[0] + p[2])/(2.0 * p[0]) - np.sqrt(D)/(2.0 * p[0]),
    -(p[0] + p[2])/(2.0 * p[2]) - np.sqrt(D)/(2.0 * p[2])
]

# Set up grid for nullclines and vector field across the wider phase space
u_vals = np.linspace(-25.0, 10.0, 500)
v_vals = np.linspace(-20.0, 20.0, 500)
U, V = np.meshgrid(u_vals, v_vals)

# System vector field components
# Intermediate function representing insulin/glucose interaction dynamics
F = -V - U * (V + 1.0) + p[3]
dU = -p[0] * U + p[1] * F + p[2] * V
dV = F

plt.figure(figsize=(10, 8))

# Plot directional stream arrows
strm = plt.streamplot(U, V, dU, dV, color='gray', density=1.2, arrowsize=1.2)
strm.lines.set_alpha(0.5)
strm.arrows.set_alpha(0.5)

# Plot nullclines where derivatives equal zero
plt.contour(U, V, dU, levels=[0], colors='blue', linewidths=2, alpha=0.8)
plt.contour(U, V, dV, levels=[0], colors='orange', linewidths=2, alpha=0.8)

# Compute saddle eigenvectors
J = gs.Jac(eq1, p)
# Eigenvalues and eigenvectors
L1, V1 = np.linalg.eig(J)
# Find stable parts
stable_idx = 0 if L1[0] < 0 else 1
# Find unstable parts
unstable_idx = 1 if stable_idx == 0 else 0
v_perpendicular = V1[:, unstable_idx]

# Scan initial conditions near the saddle point
# Loop through initial conditions near saddle point
for eps in np.linspace(-1.5, 1.5, 30):
    # Perturb initial state along eigenvalues
    u0 = np.array(eq1) + eps * v_perpendicular
    u = u0.copy()
    trajectory = [u.copy()]
    controlled = True
    
    # Simulate trajectory
    for step in range(15000):
        u = gs.RK2step(u, h, p)
        if u[1] < -100.0:
            controlled = False
            break
        trajectory.append(u.copy())
        
    trajectory = np.array(trajectory)
    color = 'g' if controlled else 'r'
    plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.6, linewidth=1.5)


# Configure axes and labels with the zoomed-out bounds
plt.xlim(-25.0, 10.0)
plt.ylim(-20.0, 20.0)
plt.title(f"Global Phase Portrait with Streamlines and Nullclines ($G = {p[3]}$)")
plt.xlabel("u'")
plt.ylabel("v'")

# Custom legend
custom_lines = [
    mlines.Line2D([], [], color='blue', lw=2, label='du/dt = 0 Nullcline'),
    mlines.Line2D([], [], color='orange', lw=2, label='dv/dt = 0 Nullcline'),
    mlines.Line2D([], [], color='gray', lw=2, label='Vector Field Streamlines'),
    mlines.Line2D([], [], color='green', lw=2, label='Successful Trajectory'),
    mlines.Line2D([], [], color='red', lw=2, label='Divergent Trajectory'),
    mlines.Line2D([], [], color='g', marker='o', linestyle='None', markersize=8, label='Stable Node (eq0)'),
    mlines.Line2D([], [], color='r', marker='o', linestyle='None', markersize=8, label='Saddle Point (eq1)')
]
plt.legend(handles=custom_lines, loc='upper left', framealpha=0.9)

plt.grid(True)
plt.savefig('global_phase_portrait_wide.png', dpi=300)
plt.show()