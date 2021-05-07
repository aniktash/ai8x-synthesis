#!/bin/sh
./quantize.py ../ai8x-training/logs/2021.05.06-174053-wl5-transform-weight/qat_best.pth.tar trained/ai85-wildlife82-chw.pth.tar --device MAX78000 -v -c networks/wildlife-chw.yaml "$@"
