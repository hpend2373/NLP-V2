#!/bin/bash
# Training script for IKEA MEPNet

# Default configuration
CONFIG="configs/train_config.yaml"
GPUS="0"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift
      shift
      ;;
    --resume)
      RESUME="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set GPU devices
export CUDA_VISIBLE_DEVICES=$GPUS

echo "Training IKEA MEPNet"
echo "Config: $CONFIG"
echo "GPUs: $GPUS"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run training
if [ -z "$RESUME" ]; then
    python scripts/train/train_ikea.py --config "$CONFIG"
else
    echo "Resuming from: $RESUME"
    python scripts/train/train_ikea.py --config "$CONFIG" --resume "$RESUME"
fi