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

@dataclass
class Robot():
    odom_cov: np.ndarray
    range_bearing_cov: np.ndarray
    loop_closure_cov: np.ndarray
    sensor_range: float
    
    def step(self, x, v, w, dt):
        """
        Integrate unicycle kinematic model for `dt` seconds with a constant
        angular velocity `w` and constant speed `v`.

        Args:
            x: state vector [px, py, th]
            v: constant speed
            w: constant angular velocity
            dt: time step
        """
        px, py, th = x
        if abs(w) < 1e-9:
            return np.array([px + v*dt*np.cos(th),
                             py + v*dt*np.sin(th),
                             th])
        r   = v / w
        dth = w * dt
        cx  = px - r*np.sin(th)
        cy  = py + r*np.cos(th)
        th2 = th + dth
        x2  = cx + r*np.sin(th2)
        y2  = cy - r*np.cos(th2)
        return np.array([x2, y2, wrap_angle(th2)])

    def add_noise(self, x, cov):
        """
        Add noise to the array `x` given a covariance matrix `cov`.
        """
        return x + np.random.multivariate_normal(np.zeros(len(x)), cov)

    def odometry_measurement(self, x1, x2):
        """
        Given a current pose `x1` and next pose `x2` compute a noisy odometry
        measurement of the relative pose.
        """
        rel_pose_true  = se2_between(x1, x2)
        return self.add_noise(rel_pose_true, self.odom_cov)

    def range_bearing_measurement(self, x, m):
        """
        Compute a noisy range-bearing measurement [r, b] given the robot's 
        current pose, `x`, and the landmark position `m`.
        """
        # dx, dy = m[0]-x[0], m[1]-x[1]
        # r = math.hypot(dx, dy)
        rel_pos = m - x[:2]
        r = np.linalg.norm(rel_pos)
        if r <= self.sensor_range:
            b = wrap_angle(np.atan2(rel_pos[1], rel_pos[0]) - x[2])
            return self.add_noise(np.array([r, b]), self.range_bearing_cov)
        else:
            return None

    def loop_closure_measurement(self, x0, xT):
        """
        Compute a noisy loop closure measurement of the relative pose between
        the poses `x0` and `xT`.
        """
        lc_true = se2_between(x0, xT)
        return self.add_noise(lc_true, self.loop_closure_cov)

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
        [ 0.0*L,  0.5*L], [ 1.0*L,  0.5*L], [ 0.5*L,  0.0*L], [ 0.5*L,  1.0*L],
        [ 0.35*L, 0.35*L], [0.65*L,0.35*L], [0.35*L,0.65*L], [0.65*L,0.65*L],
    ], dtype=float)

    return World(landmarks=landmarks, side_length=L)


def square_loop_control_sequence(world: World, v: float = 0.25, dt: float = 0.1) -> List[Tuple[float, float]]:
    """
    Compute a control sequence that follows a "square" loop through 
    the world along the sides. Each control is a vector (v, w) where
    v is the robot's speed and w is an angular rotation rate [rad/s].

    Args:
        v: speed
        dt: timestep
    """
    T_side = round(world.side_length / (dt * v))
    controls: List[Tuple[float, float]] = []
    for _ in range(4):
        for _ in range(T_side):
            controls.append((v, 0.0))
        turn_steps = max(10, int(1.0 / dt)) # ~1 s turn
        w_turn = (np.pi/2.0) / (turn_steps*dt)
        for _ in range(turn_steps):
            controls.append((0.0, w_turn))
    return controls


def plot_traj(title, series: Dict[str, np.ndarray], world: World):
    plt.figure(figsize=(6.5, 6.5))
    for name, traj in series.items():
        plt.plot(traj[:,0], traj[:,1], label=name, linewidth=1.6)
    world.plot()
    

    all_traj = np.vstack(list(series.values()))
    xmin, ymin = all_traj[:,0].min(), all_traj[:,1].min()
    xmax, ymax = all_traj[:,0].max(), all_traj[:,1].max()
    pad = 0.12 * max(xmax - xmin, ymax - ymin)
    plt.xlim(xmin - pad, xmax + pad); plt.ylim(ymin - pad, ymax + pad)

    plt.axis('equal'); plt.grid(True, alpha=0.3)
    plt.title(title); plt.legend()
    plt.show()

def simulate(robot: Robot,
             world: World,
             controls: List[Tuple[float, float]],
             dt: float):
    """
    Return GT, odom rel. measurements, landmark measurements, loop-closure measurement.
    """
    # Ground truth trajectory
    T = len(controls)
    x = np.zeros((T + 1, 3))
    for t, (v, w) in enumerate(controls):
        x[t+1] = robot.step(x[t], v, w, dt)

    # Odometry rel. measurements
    odom_meas = np.zeros((T, 3))
    for t in range(T):
        odom_meas[t] = robot.odometry_measurement(x[t], x[t+1])
        
    # Landmark range-bearing measurements
    lm_meas: List[List[Tuple[int, np.ndarray]]] = []
    for t in range(T+1):
        zt = []
        for j, m in enumerate(world.landmarks):
            z = robot.range_bearing_measurement(x[t], m)
            if z is not None:
                zt.append((j, z))
        lm_meas.append(zt)

    # Loop closure measurement
    lc_meas = robot.loop_closure_measurement(x[0], x[-1])
    return x, odom_meas, lm_meas, lc_meas

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