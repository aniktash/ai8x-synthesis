#!/bin/sh
DEVICE="MAX78000"
COMMON_ARGS="--device $DEVICE --compact-data --mexpress --timer 0 --display-checkpoint --fifo"
./ai8xize.py --verbose --log --test-dir pytorch --prefix pt-rps82 --checkpoint-file trained/ai85-rps82-chw.pth.tar --config-file networks/rps-chw.yaml --softmax --embedded-code $COMMON_ARGS "$@"

#./ai8xize.py --verbose -L --top-level cnn --test-dir tensorflow --prefix tf-rock --checkpoint-file ../ai8x-training/TensorFlow/export/rock/saved_model.onnx --config-file ./networks/rock-chw-tf.yaml --sample-input ../ai8x-training/TensorFlow/export/rock/sampledata.npy --device MAX78000 --compact-data --mexpress --embedded-code --keep-first --scale 1.0 --softmax --fifo --generate-dequantized-onnx-file "$@"