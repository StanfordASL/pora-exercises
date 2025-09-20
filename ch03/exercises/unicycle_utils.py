import typing as T
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

class State:
    """
    Define the state of the extended kinematic unicycle
    """
    def __init__(self, x: float, y: float, v: float, th: float) -> None:
        self.x = x
        self.y = y
        self.v = v
        self.th = th

    @property
    def xd(self) -> float:
        return self.v * np.cos(self.th)

    @property
    def yd(self) -> float:
        return self.v * np.sin(self.th)

def unicycle_dynamics(x: np.ndarray, t: float, u: np.ndarray, noise: np.ndarray) -> T.List[float]:
    """
    Differential equation describing the kinematic unicycle with state x = [x, y, th] and
    control u = [v, om], with assumed noise on the control inputs.
    """
    u_0 = u[0] + noise[0]
    u_1 = u[1] + noise[1]
    dxdt = [u_0 * np.cos(x[2]),
            u_0 * np.sin(x[2]),
            u_1]
    return dxdt

def simulate_unicycle(
    x_0: float,
    y_0: float,
    th_0: float,
    times: T.List[float],
    controller: T.Optional[T.Any] = None,
    open_loop_control: T.Optional[np.ndarray] = None,
    noise_scale: float = 0.
) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the unicycle model given a closed-loop controller OR a sequence of control inputs
    to appl open-loop.
    
    inputs: x_0, y_0, th_0 (floats) initial state
            times (list len N) sequence of times at which to apply control
            controller: controller object to use to compute feedback control
            open_loop_control: (np.array shape: N-1, 2) list of control inputs to apply
            noise_scale: (float) standard deviation of control noise

            if controller is provided, simulates feedback control by calling
                controller.compute_control(x,y,th,t) at each time step
            otherwise, if the array open_loop_control is specified, they are applied open loop

            (one of controller or open_loop_control must be specified)

    outputs: traj (np.array shape (N, 3)) sequence of [x,y,th] state vectors
             ctrl (np.array shape (N-1, 2)) sequence of [v, om] control vectors
    """

    feedback = False
    if controller:
        feedback = True
    elif open_loop_control is None:
        print("Either provide a controller or a sequence of open loop actions")
        raise Exception

    x = np.array([x_0, y_0, th_0])
    N = len(times)
    traj = np.zeros([N, 3])
    noise = noise_scale * np.random.randn(N, 2) # control noise
    ctrl = np.zeros([N, 2])
    for i,t in enumerate(times[:-1]):
        traj[i,:] = x

        # Compute control
        if feedback:
            v, om = controller.compute_control(x[0], x[1], x[2], t)
        elif open_loop_control is not None:
            v = open_loop_control[i,0]
            om = open_loop_control[i,1]

        ctrl[i,0] = v
        ctrl[i,1] = om

        # Apply control and simulate forward
        d_state = odeint(unicycle_dynamics, x, [t, times[i+1]], args=(ctrl[i,:], noise[i,:]))
        x = d_state[1,:]

    # Log final state
    traj[-1,:] = x

    return traj, ctrl

def plot_unicycle_traj(t, trajs, ctrls, labels, linestyles=None, alphas=None):
    plt.figure(figsize=[10,6])
    plt.subplot(1,2,1)
    for i, traj in enumerate(trajs):
        style = linestyles[i] if linestyles is not None else '-'
        alpha = alphas[i] if alphas is not None else 1
        plt.plot(traj[:,0], traj[:,1], color='k', label=labels[i], linestyle=style, alpha=alpha)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.subplot(3,2,2)
    for i, traj in enumerate(trajs):
        style = linestyles[i] if linestyles is not None else '-'
        alpha = alphas[i] if alphas is not None else 1
        plt.plot(t, traj[:,2], label=labels[i], color='g', linestyle=style, alpha=alpha)
    plt.ylabel('th')
    plt.legend()
    plt.grid(True)
    plt.xlabel('t')
    plt.subplot(3,2,4)
    for i, ctrl in enumerate(ctrls):
        style = linestyles[i] if linestyles is not None else '-'
        alpha = alphas[i] if alphas is not None else 1
        plt.plot(t[:-1], ctrl[:-1,0], label=labels[i], color='b', linestyle=style, alpha=alpha)
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel('v')
    plt.legend()
    plt.subplot(3,2,6)
    for i, ctrl in enumerate(ctrls):
        style = linestyles[i] if linestyles is not None else '-'
        alpha = alphas[i] if alphas is not None else 1
        plt.plot(t[:-1], ctrl[:-1,1], label=labels[i], color='m', linestyle=style, alpha=alpha)
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel('om')
    plt.legend()
    plt.tight_layout()