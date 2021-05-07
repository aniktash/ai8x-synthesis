#!/bin/sh
DEVICE="MAX78000"
COMMON_ARGS="--device $DEVICE --compact-data --mexpress --timer 0 --display-checkpoint --fifo"
./ai8xize.py --verbose --log --test-dir pytorch --prefix pt-wildlife82 --checkpoint-file trained/ai85-wildlife82-chw.pth.tar --config-file networks/wildlife-chw.yaml --softmax --embedded-code $COMMON_ARGS "$@"
