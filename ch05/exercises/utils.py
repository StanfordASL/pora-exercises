from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString

class Workspace2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic.
    """
    def __init__(self, width: float, height: float):
        self.lower = np.array([0, 0]) # state space lower bound
        self.upper = np.array([width, height]) # state space upper bound
        self.obstacles = [] # list of Polygons

    def set_obstacles(self, obstacles: list[Polygon]):
        self.obstacles = obstacles

    @property
    def width(self):
        return self.upper[0] - self.lower[0]

    @property
    def height(self):
        return self.upper[1] - self.lower[1]

    def in_statespace(self, x: np.ndarray):
        """
        Checks if the point is inside the valid statespace
        """
        return all(x < self.upper) and all(x > self.lower)

    def random_sample(self):
        """
        Compute a random sample point within the statespace
        """
        return self.lower + np.random.rand(2)*(self.upper - self.lower)

    def plot(self, fig_num=0, fig_size=6):
        """
        Plots the space and its obstacles.
        """
        fig = plt.figure(fig_num, figsize=[fig_size, fig_size])
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Polygon(np.array(obs.exterior.coords)))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))

class Node:
    def __init__(self, point: np.ndarray, parent: Optional[int] = None, cost: Optional[float] = float('inf')):
        self.point = point
        self.parent = parent
        self.cost = cost