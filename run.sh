#!/bin/bash

./non-ai/SVM/main.py \
  --dataset-dir archive/DataSet1 \
  --size 128 \
  --svm-kernels linear \
  --enable-noise-features \
  --enable-metadata-flags \
  --enable-patch-consistency \
  --output-csv sweep-results.csv \
  "$@"
