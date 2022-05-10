"""GNSS range measurement based edge for FGO state estimation

"""

__authors__ = "Ashwin Kanhere"
__date__ = "10 May 2022"

import numpy as np
from matplotlib import pyplot as plt

from .base_edge import BaseEdge
from ..pose.r2 import PoseR2
from ..pose.se2 import PoseSE2
from ..pose.r3 import PoseR3
from ..pose.se3 import PoseSE3


class EdgeRanging(BaseEdge):
    def calc_error(self):
        expected = np.linalg.norm(self.vertices[1].pose.position - self.vertices[0].pose.position)
        err = (self.estimate - expected).to_compact()
        return err

    def calc_jacobians(self):
        #TODO: Introduce mask for which indices are position and which are not
        #TODO: Use more general jacobian for SE(2), SE(3) and other poses 
        p0 = self.vertices[0].pose.position
        p1 = self.vertices[1].pose.position
        expected = np.linalg.norm(p1 - p0)
        j_wrt_p0 = np.dot((p0 - p1)/expected, self.vertices[0].pose.jacobian_boxplus())
        j_wrt_p1 = np.dot((p0 - p1)/expected, self.vertices[1].pose.jacobian_boxplus())
        jacobians = [j_wrt_p0, j_wrt_p1]
        return jacobians

    def plot(self, color='b'):
        """Plot the edge.

        Parameters
        ----------
        color : str
            The color that will be used to plot the edge

        """
        if plt is None:  # pragma: no cover
            raise NotImplementedError

        if isinstance(self.vertices[0].pose, (PoseR2, PoseSE2)):
            xy = np.array([v.pose.position for v in self.vertices])
            plt.plot(xy[:, 0], xy[:, 1], color=color)

        elif isinstance(self.vertices[0].pose, (PoseR3, PoseSE3)):
            xyz = np.array([v.pose.position for v in self.vertices])
            plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color)

        else:
            raise NotImplementedError