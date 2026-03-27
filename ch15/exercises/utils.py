import numpy as np
import matplotlib.pyplot as plt

def rmse(x, y):
    """Compute root mean square error."""
    diff_sq = np.square(y-x)
    mean = np.mean(diff_sq)
    return np.sqrt(mean)

def observable(A, C):
    """
    Determine whether the system x+ = Ax + Bu with measurement
    y = Cx is observable.
    """
    n = A.shape[0]
    o = C.shape[0]
    O = np.zeros((n * o, n))
    powerA = np.eye(n)
    for i in range(n):
        O[i * o:(i + 1) * o, :] = C @ powerA
        powerA = powerA @ A
    rank = np.linalg.matrix_rank(O)
    if rank == n:
        return True
    else:
        return False

def plot_car_estimation(x, mean, T):
    """Plot car state and estimate trajectories."""
    plt.figure(figsize=(12, 6))
    plt.subplot(311)
    t = T * np.arange(x.shape[1])
    plt.plot(t, x[0,:], label='true')
    plt.plot(t, mean[0,:], label='est', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()

    plt.subplot(312)
    plt.plot(t, x[1,:], label='true')
    plt.plot(t, mean[1,:], label='est', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m]')
    plt.legend()

    plt.subplot(313)
    plt.plot(t, x[2,:], label='true')
    plt.plot(t, mean[2,:], label='est', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m]')
    plt.legend()

    if x.shape[0] == 4:
        plt.figure(plt.figure(figsize=(12, 2)))
        plt.plot(t, x[3,:], label='true')
        plt.plot(t, mean[3,:], label='est', linestyle='--')
        plt.xlabel('Time [s]')
        plt.ylabel('GNSS Bias [m]')
        plt.legend()