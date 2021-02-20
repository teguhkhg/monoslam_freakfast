import cv2

class Params(object):
    def __init__(self):
        
        self.pnp_min_measurements = 10
        self.pnp_max_iterations = 10
        self.init_min_points = 10

        self.local_window_size = 10
        self.ba_max_iterations = 10

        self.min_tracked_points_ratio = 0.5

        self.lc_min_inbetween_frames = 10
        self.lc_max_inbetween_distance = 3
        self.lc_embedding_distance = 22.0
        self.lc_inliers_threshold = 15
        self.lc_inliers_ratio = 0.5
        self.lc_distance_threshold = 2
        self.lc_max_iterations = 20

        self.ground = False

        self.view_camera_size = 1

class ParamsKITTI(Params):
    def __init__(self, config="orb"):
        super().__init__()

        if config == "orb":
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=1, edgeThreshold=31)
            self.descriptor_extractor = self.feature_detector
        else:
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=1, edgeThreshold=31)
            self.descriptor_extractor = cv2.xfeatures2d.FREAK_create(
                orientationNormalized=True, scaleNormalized=True, patternScale=22.0)

        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15
        self.matching_neighborhood = 3
        self.matching_distance = 30

        self.frustum_near = 0.1
        self.frustum_far = 1000.0

        self.ground = True

        self.lc_max_inbetween_distance = 50
        self.lc_distance_threshold = 15
        self.lc_embedding_distance = 20.0

        self.view_image_width = 400
        self.view_image_height = 130
        self.view_camera_width = 0.75
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -500
        self.view_viewpoint_z = -100
        self.view_viewpoint_f = 2000