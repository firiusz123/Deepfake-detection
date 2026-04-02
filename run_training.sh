#!/bin/bash
/root/.cline/worktrees/52937/Deepfake-detection/pyenv/bin/python train.py --archive_path ./archive --mode train
/root/.cline/worktrees/52937/Deepfake-detection/pyenv/bin/python train.py --archive_path ./archive --mode test
