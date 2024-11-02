#!/bin/bash

# Log the start time of the script
echo "Starting continuous learning script at $(date)" >> /home/hamza-berrada/Desktop/cooding/SIDEHASSEL/Stock-Prediction-Models/my_tests/continuous_learning.log

# Navigate to the script directory
cd /home/hamza-berrada/Desktop/cooding/SIDEHASSEL/Stock-Prediction-Models/my_tests

# Activate virtual environment if necessary
# Uncomment and modify the following line if using a virtual environment:
conda activate airflow_env

# Run the Python script and log output
python3 continuous_learning.py >> /home/hamza-berrada/Desktop/cooding/SIDEHASSEL/Stock-Prediction-Models/my_tests/continuous_learning.log 2>&1

# Log the completion time of the script
echo "Completed continuous learning script at $(date)" >> /home/hamza-berrada/Desktop/cooding/SIDEHASSEL/Stock-Prediction-Models/my_tests/continuous_learning.log
