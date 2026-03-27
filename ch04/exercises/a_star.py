import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import Workspace2D

class AStar(object):
    """
    Class to define a motion planning problem to be solved using A*
    """

    def __init__(self, 
                 workspace: Workspace2D, 
                 x_init: tuple[float], 
                 x_goal: tuple[float], 
                 resolution: float=1.):
        """
        Initialize AStar problem.

        Args:
            workspace: 2D workspace to consider
            x_init: initial position
            x_goal: goal position
            resolution: state space resoltion
        """
        self.workspace = workspace          # workspace grid (a Workspace2D object)
        self.res = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init, self.x_goal)

        self.path = None        # the final path as a list of states

    def snap_to_grid(self, x: tuple[float]) -> tuple[float]:
        """ 
        Returns the closest point on a discrete state grid.

        Args:
            x: position

        Returns:
            Position that represents the closest point to x on the discrete state grid.
        """
        x_snap = self.res * round((x[0] - self.x_offset[0]) / self.res) + self.x_offset[0]
        y_snap = self.res * round((x[1] - self.x_offset[1]) / self.res) + self.x_offset[1]
        return (x_snap, y_snap)

    def find_best_est_cost_through(self) -> tuple[float]:
        """
        Gets the state in the open set that has the lowest estimated cost from
        start to goal that passes through the state.

        Returns:
            The state found in the open set that has the lowest lowest estimated 
            cost from start to goal that passes through the state.
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self) -> list[tuple[float]]:
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location.

        Returns:
            A list of states that go from start to goal.
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, 
                  fig_num: int=0, 
                  fig_size: int=6, 
                  show_init_label: bool=True) -> None:
        """
        Plots the path found in self.path and the obstacles.
        """
        if not self.path:
            return

        self.workspace.plot(fig_num, fig_size=fig_size)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0], solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.workspace.width, 0, self.workspace.height])

    def plot_tree(self, point_size: int=15) -> None:
        """
        Plot the tree.
        """
        open_set_segments = [(x, self.came_from[x]) for x in self.open_set if x != self.x_init]
        closed_set_segments = [(x, self.came_from[x]) for x in self.closed_set if x != self.x_init]
        for segment in open_set_segments + closed_set_segments:
            x = [segment[0][0], segment[1][0]]
            y = [segment[0][1], segment[1][1]]
            plt.plot(x, y, linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def is_free(self, x: tuple[float]) -> bool:
        """
        Checks if a give state x is free, meaning it is inside the bounds of 
        the map and is not inside any obstacle.

        Args:
            x: state array

        Returns:
            Boolean True/False
        
        """
        ##### YOUR CODE STARTS HERE #####
        # Hint: self.workspace is a Workspace2D object, take a look at its methods 
        # for what might be useful here
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######

    def distance(self, x1: tuple[float], x2: tuple[float]) -> float:
        """
        Computes the Euclidean distance between two states.

        Args:
            x1: First state
            x2: Second state

        Returns:
            Euclidean distance.
        """
        ##### YOUR CODE STARTS HERE #####
        # HINT: This should take one line.
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######

    def get_neighbors(self, x: tuple[float]) -> list[tuple[float]]:
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.res.

        Args:
            x: state

        Returns:
            List of neighbors that are free, as a list of arrays
        """
        neighbors = []

        ##### YOUR CODE STARTS HERE #####
        """
        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.res from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######
        return neighbors

    def solve(self) -> bool:
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path.

        Returns:
            True if a solution from x_init to x_goal was found.
        """

        ##### YOUR CODE STARTS HERE #####
        """
        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        raise NotImplementedError("Need to implement code here.")
        ###### YOUR CODE ENDS HERE ######