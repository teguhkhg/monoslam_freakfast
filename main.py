import argparse
import g2o
import numpy as np

from dataset import KITTIOdometry
from feature import ImageFeature
from params import ParamsKITTI
from components import Camera, Frame, StereoFrame

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="dataset path", required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()

    dataset = KITTIOdometry(args.path)
    params = ParamsKITTI()
    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, 
        dataset.cam.width, dataset.cam.height, 
        params.frustum_near, params.frustum_far)


    prev = None
    for i in range(100):
        feature = ImageFeature(dataset.reader[i], params)
        timestamp = dataset.timestamp[i]
        feature.extract()
        feature.draw_keypoints(delay=30)

        frame = Frame(i, g2o.Isometry3d(), feature, cam)
        # removed match filter for awhile to test, will add it back later to continue
        if not prev:
            prev = frame
        else:
            stereo = StereoFrame(frame, prev)
            mappoints, measurements = stereo.triangulate()
            print(len(mappoints), len(measurements))

if __name__ == "__main__":
    main()