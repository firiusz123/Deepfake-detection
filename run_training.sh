#!/bin/bash

MODE=$1
shift  # shift so "$@" contains only extra args

if [ "$MODE" = "train" ]; then
    python3 train.py --archive_path ./data/archive/DataSet1 --mode "train" "$@"
    python3 train.py --archive_path ./data/archive/DataSet1 --mode "test" "$@"

elif [ "$MODE" = "test" ]; then
    python3 train.py --archive_path ./data/archive/DataSet1 --mode "test" "$@"

else
    echo "Usage: ./run.sh [train|test] [additional args]"
fi
