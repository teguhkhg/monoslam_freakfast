import argparse

from dataset import KITTIOdometry
from feature import ImageFeature
from params import ParamsKITTI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="dataset path", required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()

    dataset = KITTIOdometry(args.path)
    params = ParamsKITTI()

    for i in range(100):
        feature = ImageFeature(dataset.reader[i], params)
        timestamp = dataset.timestamp[i]
        feature.extract()
        feature.draw_keypoints(delay=30)
        print(timestamp)

if __name__ == "__main__":
    main()