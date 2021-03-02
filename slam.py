import numpy as np

from optimization import BundleAdjustment

# current goal is to implement tracking
class Tracking(object):
    def __init__(self, params):
        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements):
        pass

