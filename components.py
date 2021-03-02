import numpy as np
import cv2
import g2o

from threading import Lock, Thread
from queue import Queue

from enum import Enum
from collections import defaultdict

from covisibility import GraphMapPoint, GraphMeasurement


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
            return np.logical_and.reduce([    # mappoint's position in current coordinates frame
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


class StereoFrame(object):
    def __init__(self, curr, prev):
        self.curr = curr
        self.prev = prev

    def triangulate(self):
        kps_curr, descs_curr, idx_curr = self.curr.get_unmatched_keypoints()
        kps_prev, descs_prev, idx_prev = self.prev.get_unmatched_keypoints()

        mappoints, matches = self.triangulate_points(
            kps_curr, descs_curr, kps_prev, descs_prev)

        measurements = []
        for mappoint, (i, j) in zip(mappoints, matches):
            meas = Measurement(
                Measurement.Source.TRIANGULATION,
                [kps_curr[i], kps_prev[j]],
                [descs_curr[i], descs_prev[j]])
            meas.mappoint = mappoint
            meas.view = self.curr.transform(mappoint.position)
            measurements.append(meas)

            self.curr.set_matched(idx_curr[i])
            self.prev.set_matched(idx_prev[j])

        return mappoints, measurements

    def triangulate_points(self, kps_curr, descs_curr, kps_prev, descs_prev):
        matches = self.curr.feature.row_match(
            kps_curr, descs_curr, kps_prev, descs_prev)
        assert len(matches) > 0

        px_curr = np.array([kps_curr[m.queryIdx].pt for m in matches])
        px_prev = np.array([kps_prev[m.trainIdx].pt for m in matches])

        points = cv2.triangulatePoints(
            self.curr.projection_matrix, 
            self.prev.projection_matrix, 
            px_curr.transpose(), 
            px_prev.transpose() 
            ).transpose()

        points = points[:, :3] / points[:, 3:]

        # can_view = np.logical_and(
        #     self.curr.can_view(points), 
        #     self.prev.can_view(points))

        mappoints = []
        matchs = []
        for i, point in enumerate(points):
            # if not can_view[i]:
            #     continue

            normal = point - self.curr.position
            normal = normal / np.linalg.norm(normal)

            color = self.curr.get_color(px_curr[i])

            mappoint = MapPoint(
                point, normal, descs_curr[matches[i].queryIdx], color)
            mappoints.append(mappoint)
            matchs.append((matches[i].queryIdx, matches[i].trainIdx))

        return mappoints, matchs


class MapPoint(GraphMapPoint):
    _id = 0
    _id_lock = Lock()

    def __init__(self, position, normal, descriptor, 
            color=np.zeros(3), 
            covariance=np.identity(3) * 1e-4):
        super().__init__()

        with MapPoint._id_lock:
            self.id = MapPoint._id
            MapPoint._id += 1

        self.position = position
        self.normal = normal
        self.descriptor = descriptor
        self.covariance = covariance
        self.color = color

        self.count = defaultdict(int)

    def update_position(self, position):
        self.position = position

    def update_normal(self, normal):
        self.normal = normal

    def update_descriptor(self, descriptor):
        self.descriptor = descriptor

    def set_color(self, color):
        self.color = color

    def is_bad(self):
        with self._lock:
            status =  (
                self.count['meas'] == 0
                or (self.count['outlier'] > 20
                    and self.count['outlier'] > self.count['inlier'])
                or (self.count['proj'] > 20
                    and self.count['proj'] > self.count['meas'] * 10))
            return status

    def increase_outlier_count(self):
        with self._lock:
            self.count['outlier'] += 1

    def increase_inlier_count(self):
        with self._lock:
            self.count['inlier'] += 1

    def increase_projection_count(self):
        with self._lock:
            self.count['proj'] += 1

    def increase_measurement_count(self):
        with self._lock:
            self.count['meas'] += 1


class Measurement(GraphMeasurement):
    Source = Enum('Measurement.Source', ['TRIANGULATION', 'TRACKING', 'REFIND'])

    def __init__(self, source, keypoints, descriptors):
        super().__init__()

        self.source = source
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.view = None

        self.xy = np.array(self.keypoints[0].pt)
        self.triangulation = (source == self.Source.TRIANGULATION)

    def get_descriptor(self, i=0):
        return self.descriptors[i]

    def get_keypoint(self, i=0):
        return self.keypoints[i]

    def get_descriptors(self):
        return self.descriptors

    def get_keypoints(self):
        return self.keypoints

    def from_triangulation(self):
        return self.triangulation

    def from_tracking(self):
        return self.source == Measurement.Source.TRACKING

    def from_refind(self):
        return self.source == Measurement.Source.REFIND

