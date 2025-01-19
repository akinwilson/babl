#!/bin/bash

echo "Pulling training and validation datasets ..."
mkdir -p inputs
gdown --id 1enHDeeAySxoNIGvSew6Y5aFxBjEVe01w --output inputs/50k.jsonl
gdown --id 1IuywHW-sjNDfMXssOimDwvvbeMaUKstq --output inputs/10k.jsonl