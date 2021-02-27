import numpy as np
import g2o

def invert_depth(x):
    return np.array([x[0], x[1], 1]) / x[2]

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()

        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        terminate = g2o.SparseOptimizerTerminateAction()
        terminate.set_gain_threshold(1e-6)
        super().add_post_iteration_action(terminate)

        self.delta = np.sqrt(5.991)
        self.aborted = False

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)
        try:
            return not self.aborted
        finally:
            self.aborted = False

    def add_pose(self, pose_id, pose, cam, fixed=False):
        pass

