import argparse

from dataset import KITTIOdometry

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="dataset path", required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()

    dataset = KITTIOdometry(args.path)

    print(len(dataset))
    print(len(dataset.timestamps))

if __name__ == "__main__":
    main()