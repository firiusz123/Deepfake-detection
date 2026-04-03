#!/bin/bash
# Any arguments passed to this script (e.g. --epochs 10) are appended to the training command.
python3 train.py --archive_path ./archive/DataSet1 --mode "train" "$@"
python3 train.py --archive_path ./archive/DataSet1 --mode "test"
