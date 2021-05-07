#!/bin/sh
#./quantize.py trained/ai85-rps-unquantized.pth.tar trained/ai85-rps-chw.pth.tar --device MAX78000 -v -c networks/rps-chw.yaml "$@"
./quantize.py trained/checkpoint.pth.tar trained/ai85-rps82-chw.pth.tar --device MAX78000 -v -c networks/rps-chw.yaml "$@"
