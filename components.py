import numpy as np
import cv2
import g2o

from threading import Lock, Thread
from queue import Queue

from enum import Enum
from collections import defaultdict

class Camera(object):
    def __init__(self, fx, fy, cx, cy, width, height, 
            frustum_near, frustum_far):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.intrinsic = np.array([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0, 0, 1]])

        self.frustum_near = frustum_near
        self.frustum_far = frustum_far

        self.width = width
        self.height = height

class Frame(object):
    def __init__(self, idx, pose, feature, cam, timestamp=None, 
            pose_covariance=np.identity(6)):
        self.idx = idx
        self.pose = pose
        self.feature = feature
        self.cam = cam
        self.timestamp = timestamp
        self.image = feature.image
        
        self.orientation = pose.orientation()
        self.position = pose.position()
        self.pose_covariance = pose_covariance

        self.transform_matrix = pose.inverse().matrix()[:3]
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))

    def can_view(self, points, ground=False, margin=20):
        points = np.transpose(points)
        (u, v), depth = self.project(self.transform(points))

        if ground:
            return np.logical_and.reduce([
                depth >= self.cam.frustum_near,
                depth <= self.cam.frustum_far,
                u >= - margin,
                u <= self.cam.width + margin])
        else:
            return np.logical_and.reduce([
                depth >= self.cam.frustum_near,
                depth <= self.cam.frustum_far,
                u >= - margin,
                u <= self.cam.width + margin,
                v >= - margin,
                v <= self.cam.height + margin])

        
    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose   
        self.orientation = self.pose.orientation()  
        self.position = self.pose.position()

        self.transform_matrix = self.pose.inverse().matrix()[:3]
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))

    def transform(self, points):
        R = self.transform_matrix[:3, :3]
        if points.ndim == 1:
            t = self.transform_matrix[:3, 3]
        else:
            t = self.transform_matrix[:3, 3:]
        return R.dot(points) + t

    def project(self, points): 
        projection = self.cam.intrinsic.dot(points / points[-1:])
        return projection[:2], points[-1]

    def find_matches(self, points, descriptors):
        points = np.transpose(points)
        proj, _ = self.project(self.transform(points))
        proj = proj.transpose()
        return self.feature.find_matches(proj, descriptors)

    def get_keypoint(self, i):
        return self.feature.get_keypoint(i)

    def get_descriptor(self, i):
        return self.feature.get_descriptor(i)

    def get_color(self, pt):
        return self.feature.get_color(pt)

    def set_matched(self, i):
        self.feature.set_matched(i)
        
    def get_unmatched_keypoints(self):
        return self.feature.get_unmatched_keypoints()

class KeyFrame(Frame):
    def __init__(self, idx, pose, feature, cam, timestamp=None, 
            pose_covariance=np.identity(6)):
        super().__init__(idx, pose, feature, cam, timestamp, pose_covariance)