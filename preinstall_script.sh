#!/bin/bash
# May need to uncomment and update to find current packages
# apt-get update

# Required for demo script! #
pip install scikit-video

# Add your desired packages for each workspace initialization
#          Add here!          #
sudo apt-get update
sudo apt-get install cuda-libraries-9-0
export LD_LIBRARY_PATH=/usr/local/cuda9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
pip install tensorflow-gpu --upgrade

