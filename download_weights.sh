#!/usr/bin/env bash
set -e

# Define URLs
PERC_URL="https://drive.google.com/file/d/16oQBicyqhpE26Vml6ncqk-hHPOTg3Xos/view?usp=drive_link"
VGG_URL="https://drive.google.com/file/d/1OS3VIUdQ3uaxsNCdPkpNEp2gFPPmN-mn/view?usp=sharing"

# Download weights if not present
if [ ! -f vgg11_mnist.pth ]; then
  echo "Downloading VGG-11 weights..."
  curl -L -o vgg11_mnist.pth "$VGG_URL"
fi

if [ ! -f perceptron_ova_mnist.pth ]; then
  echo "Downloading Perceptron weights..."
  curl -L -o perceptron_ova_mnist.pth "$PERC_URL"
fi

echo "All weights are downloaded."
