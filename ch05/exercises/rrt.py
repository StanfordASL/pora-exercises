import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from utils import Workspace2D, Node

class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""
    def __init__(self, workspace: Workspace2D, x_init, x_goal):
        self.workspace = workspace
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.V = []             # list of Nodes in the tree
        self.path = None        # the final path as a list of states

    def plot_problem(self):
        """
        Plot the workspace, start, and goal points
        """
        self.workspace.plot()
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)

    def is_free_motion(self, x1: np.ndarray, x2: np.ndarray):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRT")

    def nearest_neighbor(self, x):
        """
        Given a query state x, returns the index of self.V such that the steering 
        distance (subject to robot dynamics) from V[i] to x is minimized.

        Inputs:
            x - query state
        Output:
            Integer index of nearest point in self.V to x
        """
        raise NotImplementedError("nearest_neighbor must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")

    def solve(self, eps, max_iters=1000, goal_bias=0.05, shortcut=False):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """
        self.V = [Node(self.x_init)] # reset the list of nodes

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - self.V: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of Nodes
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.

        ########## Code starts here ##########
        ## Hints:
        #   - use the helper functions nearest_neighbor, steer_towards, and is_free_motion
        #   - the order in which you pass in arguments to steer_towards and is_free_motion is important

        ########## Code ends here ##########

        # Plot the results
        self.plot_problem()
        self.plot_tree(color="blue", linewidth=.5, label="RRT tree", alpha=0.5)
        if success:
            if shortcut:
                self.plot_path(color="purple", linewidth=2, label="Original solution path")
                self.shortcut_path()
                self.plot_path(color="green", linewidth=2, label="Shortcut solution path")
            else:
                self.plot_path(color="green", linewidth=2, label="Solution path")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
            plt.scatter([node.point[0] for node in self.V], [node.point[1] for node in self.V])

        else:
            print("Solution not found!")

        return success

    def solve_star(self, eps, max_iters=1000, goal_bias=0.05, shortcut=False):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """
        self.V = [Node(self.x_init)] # reset the list of nodes

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - self.V: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of Nodes
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.

        ########## Code starts here ##########
        ## Hints:
        #   - use the helper functions nearest_neighbor, steer_towards, and is_free_motion
        #   - the order in which you pass in arguments to steer_towards and is_free_motion is important

        ########## Code ends here ##########

        # Plot the results
        self.plot_problem()
        self.plot_tree(color="blue", linewidth=.5, label="RRT tree", alpha=0.5)
        if success:
            if shortcut:
                self.plot_path(color="purple", linewidth=2, label="Original solution path")
                self.shortcut_path()
                self.plot_path(color="green", linewidth=2, label="Shortcut solution path")
            else:
                self.plot_path(color="green", linewidth=2, label="Solution path")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
            plt.scatter([node.point[0] for node in self.V], [node.point[1] for node in self.V])

        else:
            print("Solution not found!")

        return success

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.

        Input:
            None
        Output:
            None, but should modify self.path
        """
        ########## Code starts here ##########
        raise NotImplementedError("Implement before use")
        ########## Code ends here ##########

class GeometricRRT(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """
    def nearest_neighbor(self, x):
        # Consult function specification in parent (RRT) class.
        ########## Code starts here ##########
        raise NotImplementedError("Implement before use")
        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        # Consult function specification in parent (RRT) class.
        ########## Code starts here ##########
        # Hint: This should take 1-4 line.
        raise NotImplementedError("Implement before use")
        ########## Code ends here ##########
        pass

    def is_free_motion(self, x1, x2):
        line = LineString([x1, x2])
        for obstacle in self.workspace.obstacles:
            if line.intersects(obstacle):
                return False
        return True

    def plot_tree(self, **kwargs):
        for node in self.V:
            if node.parent is not None:
                parent_node = self.V[node.parent]
                x = [node.point[0], parent_node.point[0]]
                y = [node.point[1], parent_node.point[1]]
                plt.plot(x, y, linewidth=1, color="blue", alpha=0.2)

    def plot_path(self, **kwargs):
        if len(self.path) > 1:
            x = [node.point[0] for node in self.path]
            y = [node.point[1] for node in self.path]
            plt.plot(x, y, **kwargs)
