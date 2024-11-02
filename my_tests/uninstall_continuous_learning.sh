#!/bin/bash

# Define paths and filenames
SCRIPT_DIR=$(pwd)
RUN_SCRIPT="run_continuous_learning.sh"
LOG_FILE="continuous_learning.log"

# Step 1: Remove the cron job for run_continuous_learning.sh
echo "Removing cron job for $RUN_SCRIPT..."

# Use crontab to remove any lines that reference run_continuous_learning.sh
crontab -l | grep -v "$SCRIPT_DIR/$RUN_SCRIPT" | crontab -

# Confirm cron job removal
if [ $? -eq 0 ]; then
    echo "Cron job for $RUN_SCRIPT removed successfully."
else
    echo "Failed to remove cron job. Please check your crontab manually."
fi

# Step 2: Remove the run_continuous_learning.sh script
if [ -f "$SCRIPT_DIR/$RUN_SCRIPT" ]; then
    echo "Removing $RUN_SCRIPT..."
    rm "$SCRIPT_DIR/$RUN_SCRIPT"
    echo "$RUN_SCRIPT removed successfully."
else
    echo "$RUN_SCRIPT not found. Skipping removal of the script."
fi

# Step 3: Optional - Remove the log file
if [ -f "$SCRIPT_DIR/$LOG_FILE" ]; then
    read -p "Do you want to remove the log file ($LOG_FILE)? (y/n): " choice
    if [ "$choice" == "y" ] || [ "$choice" == "Y" ]; then
        rm "$SCRIPT_DIR/$LOG_FILE"
        echo "$LOG_FILE removed successfully."
    else
        echo "Log file ($LOG_FILE) retained."
    fi
else
    echo "$LOG_FILE not found. Skipping removal of the log file."
fi

# Completion message
echo "Uninstallation complete."
