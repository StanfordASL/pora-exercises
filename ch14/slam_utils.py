from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# SE(2) helpers
# -----------------------------
def wrap_angle(a: float) -> float:
    """
    Wrap angle to be in the range [-pi, pi).
    """
    return (a + np.pi) % (2*np.pi) - np.pi

def rotation(th: float) -> np.ndarray:
    """
    Compute rotation matrix for angle `th`, such that
    R*x will rotate the vector x by the angle `th`.
    """
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],
                     [s,  c]])

def se2_compose(a, b):
    """
    Compose the poses `a` and `b`. 

    If `a` is the robot's pose, and `b` is a relative transformation,
    this computes a new pose in the same coordinate frame of `a`.
    """
    Ra = rotation(a[2])
    t = a[:2] + Ra @ b[:2]
    return np.array([t[0], t[1], wrap_angle(a[2] + b[2])])

def se2_inverse(p):
    """
    Compute inverse transformation of the pose `p`.
    """
    Rp = rotation(p[2])
    t = -Rp.T @ p[:2]
    return np.array([t[0], t[1], -p[2]])

def se2_between(a, b):
    """
    Compute relative transformation between poses `a` and `b`.
    """
    return se2_compose(se2_inverse(a), b)

def se2_rel_residual(a, b):
    """
    Residual where translation error is expressed in the *measurement frame*;
    angle is wrapped difference. This stabilizes GN.
    r = [ Rz(-meas_th) * (pred_xy - meas_xy) ; wrap(pred_th - meas_th) ]
    """
    dxy = a[:2] - b[:2]
    r_xy = rotation(-b[2]) @ dxy
    r_th = wrap_angle(a[2] - b[2])
    return np.array([r_xy[0], r_xy[1], r_th])

@dataclass
class World:
    """
    Create a simple square 2D toy world for a robot to navigate in that is
    contains a set of landmarks.

    Args:
        landmarks: array of shape (num_landmarks, 2) giving the 2D position of each landmark
        side_length: the length of the World's sides
    """
    landmarks: np.ndarray
    side_length: float

    def plot(self):
        plt.scatter(self.landmarks[:,0], self.landmarks[:,1], marker='x', alpha=0.5, linewidths=1.0)

    def num_landmarks(self):
        return self.landmarks.shape[0]


def make_simple_world():
    # Dense-ish ring of landmarks
    L = 5
    landmarks = np.array([
        [ 0.2*L,  0.2*L], [ 0.8*L,  0.2*L], [ 0.2*L,  0.8*L], [ 0.8*L,  0.8*L],
        [-0.3*L,  0.5*L], [ 0.5*L, -0.3*L], [ 1.3*L,  0.5*L], [ 0.5*L,  1.3*L],
        [ 0.1*L,  0.5*L], [ 1.1*L,  0.5*L], [ 0.5*L,  0.1*L], [ 0.5*L,  1.1*L],
        [ 0.35*L, 0.35*L], [0.65*L,0.35*L], [0.35*L,0.65*L], [0.65*L,0.65*L],
    ], dtype=float)

    return World(landmarks=landmarks, side_length=L)


def square_loop_control_sequence(world: World, v: float=0.25, dt: float=0.1) -> np.ndarray:
    """
    Compute a control sequence that follows a "square" loop through 
    the world along the sides. Each control is a vector (v, w) where
    v is the robot's speed and w is an angular rotation rate [rad/s].

    Args:
        v: speed
        dt: timestep
    """
    side_steps = round(world.side_length / (dt * v))
    turn_steps = max(10, int(1.0 / dt)) # ~1 s turn
    controls = []
    for _ in range(4):
        for _ in range(side_steps):
            controls.append([v, 0.0])
        w_turn = (np.pi/2.0) / (turn_steps*dt)
        for _ in range(turn_steps):
            controls.append([0.0, w_turn])
    return np.asarray(controls)


def plot_traj(title, series: Dict[str, np.ndarray], worlds: List[World]=None):
    plt.figure(figsize=(6.5, 6.5))
    for name, traj in series.items():
        plt.plot(traj[:,0], traj[:,1], label=name, linewidth=1.6)
    if worlds is not None:
        for world in worlds:
            world.plot()
    

    all_traj = np.vstack(list(series.values()))
    xmin, ymin = all_traj[:,0].min(), all_traj[:,1].min()
    xmax, ymax = all_traj[:,0].max(), all_traj[:,1].max()
    pad = 0.12 * max(xmax - xmin, ymax - ymin)
    plt.xlim(xmin - pad, xmax + pad); plt.ylim(ymin - pad, ymax + pad)

    plt.axis('equal'); plt.grid(True, alpha=0.3)
    plt.title(title); plt.legend()
    plt.show()

def absolute_trajectory_rmse(x: np.ndarray, x_est: np.ndarray) -> float:
    """
    Compute root mean squared error of the trajectory `x_est` with
    respect to the true `x`.
    """
    # simple translation alignment on first pose (x,y)
    d_est = x_est[:, :2] - x_est[0, :2]
    d  = x[:,  :2] - x[0,  :2]
    err   = d_est - d
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))