import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are

# System Parameters
m = 0.2       # Mass of the pendulum (kg)
m_measured = 0.2 # Measured mass of the pendulum
M = 1.0      # Mass of the cart (kg)
L = 0.5       # Length to pendulum center of mass (m)
L_measured = 0.5
g = 9.81      # Gravitational acceleration (m/s^2)
b = 0.0       # Cart friction coefficient
c = 0.0       # Pendulum friction coefficient

# Full Nonlinear Dynamics
def cart_pendulum_dynamics(t, state, K):
    """
    Computes the time derivative of the state for the nonlinear inverted pendulum system.
    state: [x, x_dot, theta, theta_dot] - cart position, velocity, angle, angular velocity
    K: LQR gain matrix for feedback control
    """
    x, x_dot, theta, theta_dot = state
    
    # Define the control input u (feedback control law: u = -Kx)
    u = -K @ state
    
    # Intermediate calculations
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = M + m * sin_theta**2
    
    # Equations of motion
    x_ddot = (- m * g * sin_theta * cos_theta - m * L * theta_dot**2 * sin_theta + u - b * x_dot) / denominator
    theta_ddot = (x_ddot * cos_theta + g * sin_theta - c * theta_dot) / L
    
    # Return time derivative of the state
    return [x_dot, x_ddot[0], theta_dot, theta_ddot[0]]

# Visualization Function
def animate_cart_pendulum(t, x, theta, L):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.2, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('Cart Position')
    ax.set_ylabel('Height')
    ax.set_title('Cart-Pendulum System Visualization')

    # Cart and pendulum visualization elements
    cart_width = 0.4
    cart_height = 0.2
    pendulum_line, = ax.plot([], [], 'o-', lw=4, color='blue')  # Pendulum line
    cart_rect = plt.Rectangle((0, 0), cart_width, cart_height, color='gray')  # Cart rectangle
    ax.add_patch(cart_rect)

    def init():
        pendulum_line.set_data([], [])
        cart_rect.set_xy((-cart_width / 2, -cart_height / 2))
        return pendulum_line, cart_rect

    def update(frame):
        cart_x = x[frame]
        pendulum_x = cart_x + L * np.sin(theta[frame])
        pendulum_y = L * np.cos(theta[frame])

        # Update cart position
        cart_rect.set_xy((cart_x - cart_width / 2, -cart_height / 2))

        # Update pendulum position
        pendulum_line.set_data([cart_x, pendulum_x], [0, pendulum_y])

        return pendulum_line, cart_rect

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=20)
    plt.show()

# LQR Controller Design
def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

# Main Code
if __name__ == "__main__":   

    # Linearized system matrices
    A = np.array([
        [0, 1, 0, 0],
        [0, -b / M, -m_measured * g / M, 0],
        [0, 0, 0, 1],
        [0, 0, (M + m_measured) * g / (M * L_measured), -c / (M * L_measured)]
    ])

    B = np.array([
        [0],
        [1 / M],
        [0],
        [1 / (M * L_measured)]
    ])

    # LQR cost matrices
    Q = np.diag([10, 0, 100, 0])
    R = np.array([[1]])

    # Compute LQR gain
    K = lqr(A, B, Q, R)
    # K = np.array([[0, 0, 0, 0]])
    print("LQR Gain Matrix K:", K)

    # Compute the poles of the controlled and uncontrolled system
    uncontrolled_poles = np.linalg.eigvals(A)
    controlled_poles = np.linalg.eigvals(A - B @ K)

    # Display eigenvalues
    print("\nEigenvalues of the uncontrolled system (Poles):")
    print(uncontrolled_poles)
    print("\nEigenvalues of the controlled system (Poles):")
    print(controlled_poles)

    # Initial conditions: [x, x_dot, theta, theta_dot]
    theta0 = np.pi / 6 # Initial pendulum angle deviation (30 degrees)
    x_0 = 0.0
    y0 = [x_0, 0.0, theta0, 0.0]
    
    # Time span for simulation
    t_span = (0, 20)
    
    # Simulate the system
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    solution = solve_ivp(cart_pendulum_dynamics, t_span, y0, t_eval=t_eval, args=(K,))
    
    # Extract results
    t = solution.t
    x, x_dot, theta, theta_dot = solution.y

    # Compute the control input u over time
    u = np.array([-K @ state for state in np.array([x, x_dot, theta, theta_dot]).T])  # Compute u for all time steps
    # Animate the results
    animate_cart_pendulum(t, x, theta, L)

    # Plot results
    plt.figure(figsize=(16, 12))

    plt.subplot(3, 2, 1)
    plt.plot(t, x, label="Cart Position (x)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(t, x_dot, label="Cart Velocity (x_dot)")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(t, theta, label="Pendulum Angle (theta)")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(t, theta_dot, label="Pendulum Angular Velocity (theta_dot)")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(t, u, label='Control Input u(t)')
    plt.xlabel("Time [s]")
    plt.ylabel("Control Input u [N]")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.scatter(uncontrolled_poles.real, uncontrolled_poles.imag, color='red', label='Uncontrolled Poles', s=100, marker='x')
    plt.scatter(controlled_poles.real, controlled_poles.imag, color='blue', label='Controlled Poles', s=100, marker='o')
    plt.axvline(0, color='k', linestyle='--', linewidth=1)  # Stability boundary (Real = 0)
    plt.grid()
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.show()
