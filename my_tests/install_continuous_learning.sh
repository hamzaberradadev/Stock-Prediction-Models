#!/bin/bash

# Define paths and filenames
SCRIPT_DIR=$(pwd)
PYTHON_SCRIPT="continuous_learning.py"
RUN_SCRIPT="run_continuous_learning.sh"
LOG_FILE="continuous_learning.log"

# Step 1: Check if continuous_learning.py exists
if [ ! -f "$SCRIPT_DIR/$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found in $SCRIPT_DIR."
    echo "Please ensure $PYTHON_SCRIPT is in the current directory and try again."
    exit 1
fi

# Step 2: Create the shell script to run continuous_learning.py
echo "Creating the shell script $RUN_SCRIPT..."

cat <<EOL > "$SCRIPT_DIR/$RUN_SCRIPT"
#!/bin/bash

# Log the start time of the script
echo "Starting continuous learning script at \$(date)" >> $SCRIPT_DIR/$LOG_FILE

# Navigate to the script directory
cd $SCRIPT_DIR

# Activate virtual environment if necessary
# Uncomment and modify the following line if using a virtual environment:
conda activate airflow_env

# Run the Python script and log output
python3 $PYTHON_SCRIPT >> $SCRIPT_DIR/$LOG_FILE 2>&1

# Log the completion time of the script
echo "Completed continuous learning script at \$(date)" >> $SCRIPT_DIR/$LOG_FILE
EOL

# Step 3: Make the shell script executable
echo "Setting executable permissions for $RUN_SCRIPT..."
chmod +x "$SCRIPT_DIR/$RUN_SCRIPT"

# Step 4: Set up the cron job to run the script weekly (every Monday at 2:00 AM)
echo "Setting up a weekly cron job for $RUN_SCRIPT..."

(crontab -l 2>/dev/null; echo "0 2 * * 1 $SCRIPT_DIR/$RUN_SCRIPT") | crontab -

# Confirm cron job setup
if [ $? -eq 0 ]; then
    echo "Cron job scheduled successfully. $RUN_SCRIPT will run weekly on Mondays at 2:00 AM."
else
    echo "Failed to set up cron job. Please check your crontab settings manually."
fi

# Step 5: Provide final instructions
echo "Installation complete."
echo "You can check $LOG_FILE for output logs after each scheduled run."
echo "To run the script immediately, use: $SCRIPT_DIR/$RUN_SCRIPT"
