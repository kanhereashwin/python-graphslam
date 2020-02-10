"""Unit tests for the graph.py module.

"""


import unittest

import numpy as np

from graphslam.vertex import Vertex
from graphslam.edge.base_edge import BaseEdge
from graphslam.edge.edge_odometry import OdometryEdge
from graphslam.pose.r2 import PoseR2
from graphslam.pose.se2 import PoseSE2


class SimpleEdge(BaseEdge):
    """A simple edge class for testing.

    """
    def calc_error(self):
        """A simple "error" method."""
        return len(self.vertices)


class TestBaseEdge(unittest.TestCase):
    """Tests for the ``BaseEdge`` class.

    """

    def test_constructor(self):
        """Test that a ``BaseEdge`` object can be created.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = BaseEdge([v], 1, 0)

        self.assertEqual(e.vertices[0].vertex_id, 0)
        self.assertEqual(e.information, 1)

    def test_calc_error(self):
        """Test that the ``calc_error`` method is not implemented.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = BaseEdge([v], 1, 0)

        with self.assertRaises(NotImplementedError):
            _ = e.calc_error()

    def test_calc_chi2(self):
        """Test that the ``calc_chi2`` method works as expected.

        """
        p = PoseSE2([0, 0], 0)
        v = Vertex(0, p)
        e = SimpleEdge([v], 1, 0)

        self.assertEqual(e.calc_chi2(), 1)

    def test_calc_jacobians(self):
        """Test that the ``calc_jacobians`` method works as expected.

        """
        p1 = PoseR2([1, 2])
        p2 = PoseR2([3, 4])

        v1 = Vertex(0, p1)
        v2 = Vertex(0, p2)

        e = OdometryEdge([v1, v2], np.eye(2), 0)

        jacobians = e.calc_jacobians()

        self.assertAlmostEqual(np.linalg.norm(jacobians[0] - np.eye(2)), 0.)
        self.assertAlmostEqual(np.linalg.norm(jacobians[1] + np.eye(2)), 0.)

    def test_calc_chi2_gradient_hessian(self):
        """Test that the ``calc_chi2_gradient_hessian`` method works as expected.

        """
        p1 = PoseR2([1, 3])
        p2 = PoseR2([2, 4])

        v1 = Vertex(0, p1)
        v2 = Vertex(0, p2)

        e = OdometryEdge([v1, v2], np.eye(2), 0)

        chi2, gradient, hessian = e.calc_chi2_gradient_hessian()

        self.assertEqual(chi2, 2.)

        self.assertAlmostEqual(np.linalg.norm(gradient[0] + np.ones(2)), 0.)
        self.assertAlmostEqual(np.linalg.norm(gradient[1] - np.ones(2)), 0.)

        self.assertAlmostEqual(np.linalg.norm(hessian[0] - np.eye(2)), 0.)
        self.assertAlmostEqual(np.linalg.norm(hessian[1] + np.eye(2)), 0.)
        self.assertAlmostEqual(np.linalg.norm(hessian[2] - np.eye(2)), 0.)


if __name__ == '__main__':
    unittest.main()
