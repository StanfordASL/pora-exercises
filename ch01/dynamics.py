class Dynamics():
  def __init__(self):
    super().__init__()
    self.Δt = 0.01 # timestep for integration
    self.noisy = True # whether the dynamics are noisy

  def feed_forward(self, state, control):
    """
    Compute the next state at time t + Δt given the state at time t and the control input.

    Parameters
    ----------
    state : state at time t
    control : control at time t

    Returns
    -------
    next_state : state at time t + Δt
    """
    raise NotImplementedError("Calling abstract function")

  def rollout(self, state_init, control_traj, num_rollouts):
    """
    Compute a number of state trajectories (rollouts) starting from an initial state given a control sequence.

    Parameters
    ----------
    state_init : initial state
    control_traj : control sequence
    num_rollouts : number of rollouts to compute

    Returns
    -------
    set of rollouts starting from the initial and applying the given control sequence
    """
    raise NotImplementedError("Calling abstract function")
