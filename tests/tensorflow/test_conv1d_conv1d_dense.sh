#!/bin/sh
./ai8xize.py --verbose -L --top-level cnn --test-dir tests/tensorflow --prefix conv1d-conv1d-dense --checkpoint-file ../ai8x-training/TensorFlow/test/FusedConv1D_Conv1D_Dense/saved_model/saved_model.onnx --config-file tests/tensorflow/simple-conv1d-conv1d-dense.yaml --sample-input ../ai8x-training/TensorFlow/test/FusedConv1D_Conv1D_Dense/saved_model/input_sample_7x9.npy --device 85 --compact-data --mexpress --unload --embedded-code --scale 1.0 "$@"
