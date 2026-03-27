from typing import TypeVar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation
from scipy.integrate import odeint
import jax.numpy as jnp

class CartPole(object):
    def __init__(self, mp: float, mc: float, L: float, g: float):
        """
        Initialize cart-pole physical properties

        Args:
            mp: pendulum mass [kg]
            mc: cart mass [kg]
            L: pendulum length [m]
            g: acceleration due to gravity [m/s^2]
        """
        self.mp = mp
        self.mc = mc
        self.L = L
        self.g = g

def cartpole_dynamics(s: np.ndarray, u: np.ndarray, cartpole: CartPole) -> np.ndarray:
    """
    Compute the cart-pole state derivative

    Args:
        s (np.ndarray): The cartpole state: [x, theta, x_dot, theta_dot], shape (n,)
        u (np.ndarray): The cartpole control: [F_x], shape (m,)
        cartpole: object storing cartpole params

    Returns:
        np.ndarray: The state derivative, shape (n,)
    """
    mp, mc, L, g = cartpole.mp, cartpole.mc, cartpole.L, cartpole.g
    x, θ, dx, dθ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    h = mc + mp * (sinθ**2)
    ds = jnp.array(
        [
            dx,
            dθ,
            (mp * sinθ * (L * (dθ**2) + g * cosθ) + u[0]) / h,
            -((mc + mp) * g * sinθ + mp * L * (dθ**2) * sinθ * cosθ + u[0] * cosθ)
            / (h * L),
        ]
    )
    return ds

ControllerType = TypeVar('ControllerType')
def simulate_cartpole(
    cartpole: CartPole,
    t: np.ndarray, 
    s0: np.ndarray, 
    controller: ControllerType,
    closed_loop: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the cartpole

    Args:
        cartpole: Cartpole object storing params
        t (np.ndarray): Evaluation times, shape (num_timesteps,)
        s0 (np.ndarray): Initial state, shape (n,)
        controller: Controller with function compute_control(k, s, closed_loop) where
                    k is the time step, s is the state at time k, which outputs the
                    control u for this state and time.


    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of:
            np.ndarray: The state history, shape (num_timesteps, n)
            np.ndarray: The control history, shape (num_timesteps-1, m)
    """
    s = np.zeros((t.size, 4))
    u = np.zeros((t.size - 1, 1))
    s[0] = s0
    for k in range(t.size - 1):
        u[k] = controller.compute_control(k, s[k], closed_loop=closed_loop)
        s[k+1] = odeint(lambda s, t, u: cartpole_dynamics(s, u, cartpole), s[k], t[k:k+2], (u[k],))[1]
    return s, u


def animate_cartpole(t, x, θ):
    """
    Animate the cart-pole system from given position data.

    All arguments are assumed to be 1-D NumPy arrays, where `x` and `θ` are the
    degrees of freedom of the cart-pole over time `t`.
    """
    # Geometry
    cart_width = 2.
    cart_height = 1.
    wheel_radius = 0.3
    wheel_sep = 1.
    pole_length = 5.
    mass_radius = 0.25

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_min, x_max = np.min(x) - 1.1*pole_length, np.max(x) + 1.1*pole_length
    y_min = -pole_length
    y_max = 1.1*(wheel_radius + cart_height + pole_length)
    ax.plot([x_min, x_max], [0., 0.], '-', linewidth=1, color='k')[0]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_yticks([])
    ax.set_aspect(1.)

    # Artists
    cart = mpatches.FancyBboxPatch((0., 0.), cart_width, cart_height,
                                   facecolor='tab:blue', edgecolor='k',
                                   boxstyle='Round,pad=0.,rounding_size=0.05')
    wheel_left = mpatches.Circle((0., 0.), wheel_radius, color='k')
    wheel_right = mpatches.Circle((0., 0.), wheel_radius, color='k')
    mass = mpatches.Circle((0., 0.), mass_radius, color='k')
    pole = ax.plot([], [], '-', linewidth=3, color='k')[0]
    trace = ax.plot([], [], '--', linewidth=2, color='tab:orange')[0]
    timestamp = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    ax.add_patch(cart)
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)
    ax.add_patch(mass)

    def animate(k, t, x, θ):
        # Geometry
        cart_corner = np.array([x[k] - cart_width/2, wheel_radius])
        wheel_left_center = np.array([x[k] - wheel_sep/2, wheel_radius])
        wheel_right_center = np.array([x[k] + wheel_sep/2, wheel_radius])
        pole_start = np.array([x[k], wheel_radius + cart_height])
        pole_end = pole_start + pole_length*np.array([np.sin(θ[k]),
                                                      -np.cos(θ[k])])

        # Cart
        cart.set_x(cart_corner[0])
        cart.set_y(cart_corner[1])

        # Wheels
        wheel_left.set_center(wheel_left_center)
        wheel_right.set_center(wheel_right_center)

        # Pendulum
        pole.set_data([pole_start[0], pole_end[0]],
                      [pole_start[1], pole_end[1]])
        mass.set_center(pole_end)
        mass_x = x[:k+1] + pole_length*np.sin(θ[:k+1])
        mass_y = wheel_radius + cart_height - pole_length*np.cos(θ[:k+1])
        trace.set_data(mass_x, mass_y)

        # Time-stamp
        timestamp.set_text('t = {:.1f} s'.format(t[k]))

        artists = (cart, wheel_left, wheel_right, pole, mass, trace, timestamp)
        return artists

    dt = t[1] - t[0]
    ani = animation.FuncAnimation(fig, animate, t.size, fargs=(t, x, θ),
                                  interval=dt*1000, blit=True)
    try:
        get_ipython()
        from IPython.display import HTML
        ani = HTML(ani.to_html5_video())
    except (NameError, ImportError):
        raise RuntimeError("Requires this code to be run in Jupyter notebook.")
    plt.close(fig)
    return ani

def plot_state_and_control_history(
    s: np.ndarray, u: np.ndarray, t: np.ndarray, s_ref: np.ndarray, name: str
) -> None:
    """
    Helper function for cartpole visualization

    Args:
        s (np.ndarray): State history, shape (num_timesteps, n)
        u (np.ndarray): Control history, shape (num_timesteps, m)
        t (np.ndarray): Times, shape (num_timesteps,)
        s_ref (np.ndarray): Reference state s_bar, evaluated at each time t. Shape (num_timesteps, n)
        name (str): Filename prefix for saving figures
    """
    n = s.shape[1]
    m = u.shape[1]
    N_u = u.shape[0]
    fig, axes = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
    plt.subplots_adjust(wspace=0.35)
    labels_s = (r"$x(t)$", r"$\theta(t)$", r"$\dot{x}(t)$", r"$\dot{\theta}(t)$")
    labels_u = (r"$u(t)$",)
    for i in range(n):
        axes[i].plot(t, s[:, i])
        axes[i].plot(t, s_ref[:, i], "--")
        axes[i].set_xlabel(r"$t$")
        axes[i].set_ylabel(labels_s[i])
    for i in range(m):
        axes[n + i].plot(t[:N_u], u[:, i])
        axes[n + i].set_xlabel(r"$t$")
        axes[n + i].set_ylabel(labels_u[i])
    plt.show()