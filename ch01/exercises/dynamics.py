import numpy as np

class Dynamics():
  def __init__(self):
    super().__init__()
    self.dt = 0.01 # timestep for integration
    self.noisy = True # whether the dynamics are noisy

  def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Compute the next state at time t + Δt given the state at 
    time t and the control input.

    Args:
      x: state at time t
      u: control at time t

    Returns:
      The next state, at time t + Δt.
    """
    raise NotImplementedError("Calling abstract function")

  def rollout(self, x0: np.ndarray, u_sequence: np.ndarray, num_rollouts: int) -> np.ndarray:
    """
    Compute a number of state trajectories (rollouts) starting 
    from an initial state given a control sequence.

    Args:
      x0: initial state
      u_sequence: control sequence
      num_rollouts: number of rollouts to compute

    Returns:
      Set of rollouts starting from the initial and applying 
      the given control sequence.
    """
    raise NotImplementedError("Calling abstract function")
