#!/pyenv/bin/python3
import argparse
from ML.cnn_baseline.pipeline import run_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive_path', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    run_pipeline(args, mode=args.mode)
