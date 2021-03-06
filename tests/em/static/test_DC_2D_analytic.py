from __future__ import print_function
import unittest
from SimPEG import Mesh, Utils, EM, SolverLU
import numpy as np
import SimPEG.EM.Static.DC as DC
import matplotlib.pyplot as plt

class DCProblemAnalyticTests_PDP(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        x = np.linspace(-135, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        # A1loc = np.r_[-130, 0.]
        rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
        data_anal = EM.Analytics.DCAnalytic_Pole_Dipole(
            np.r_[A0loc, 0.], rxloc, sighalf, earth_type="halfspace"
        )

        rx = DC.Rx.Dipole_ky(M, N)
        src0 = DC.Src.Pole([rx], A0loc)
        survey = DC.Survey_ky([src0])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_anal = data_anal

        try:
            from pymatsolver import Pardiso
            self.Solver = Pardiso
        except ImportError:
            self.Solver = SolverLU

    def test_Problem3D_N(self):

        problem = DC.Problem2D_N(self.mesh, sigma=self.sigma)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_anal) / self.data_anal)**2 /
            self.data_anal.size
        )
        if err < 0.05:
            passed = True
            print(">> DC analytic test for Problem3D_N is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_N is failed")
        self.assertTrue(passed)

    def test_Problem3D_CC(self):
        problem = DC.Problem2D_CC(self.mesh, sigma=self.sigma)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_anal)/self.data_anal)**2 /
            self.data_anal.size
        )
        if err < 0.05:
            passed = True
            print(">> DC analytic test for Problem3D_CC is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_CC is failed")
        self.assertTrue(passed)

# This does not work well.
class DCProblemAnalyticTests_DPP(unittest.TestCase):

    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hy], x0="CN")
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        x = np.linspace(0, 250., 20)
        M = Utils.ndgrid(x-12.5, np.r_[0.])
        N = Utils.ndgrid(x+12.5, np.r_[0.])
        A0loc = np.r_[-150, 0.]
        A1loc = np.r_[-130, 0.]
        rxloc = np.c_[M, np.zeros(20)]
        data_anal = EM.Analytics.DCAnalytic_Dipole_Pole(
                    [np.r_[A0loc, 0.],np.r_[A1loc, 0.]],
                    rxloc, sighalf, earth_type="halfspace")

        rx = DC.Rx.Pole_ky(M)
        src0 = DC.Src.Dipole([rx], A0loc, A1loc)
        survey = DC.Survey_ky([src0])

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.data_anal = data_anal

        try:
            from pymatsolver import PardisoSolver
            self.Solver = PardisoSolver
        except ImportError:
            self.Solver = SolverLU

    def test_Problem3D_N(self):

        problem = DC.Problem2D_N(self.mesh, sigma=self.sigma)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_anal) / self.data_anal)**2 /
            self.data_anal.size
        )
        if err < 0.3:
            passed = True
            print(">> DC analytic test for Problem3D_N is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_N is failed")
            print (err)
        self.assertTrue(passed)

    def test_Problem3D_CC(self):
        problem = DC.Problem2D_CC(self.mesh, sigma=self.sigma)
        problem.Solver = self.Solver
        problem.pair(self.survey)
        data = self.survey.dpred()
        err = (
            np.linalg.norm((data-self.data_anal)/self.data_anal)**2 /
            self.data_anal.size
        )
        if err < 0.3:
            passed = True
            print(">> DC analytic test for Problem3D_CC is passed")
        else:
            passed = False
            print(">> DC analytic test for Problem3D_CC is failed")
            print (err)
            plt.plot(data)
            plt.plot(self.data_anal)
            plt.show()
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
