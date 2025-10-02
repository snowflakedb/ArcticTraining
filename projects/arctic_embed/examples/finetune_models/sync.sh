#!/bin/bash

# This script continuously synchronizes a local directory to an AWS S3 bucket
# at a set interval.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---

# The local source directory.
SOURCE_DIR="/notebook/ArcticTraining/projects/arctic_embed/examples/finetune_models/checkpoints"

# The destination S3 bucket and path.
S3_DESTINATION="s3://ml-dev-sfc-or-dev-misc1-k8s/cortexsearch/pyu/arctic_training_checkpoints"

# The sync interval in seconds (30 minutes = 30 * 60 = 1800 seconds).
SYNC_INTERVAL_SECONDS=1800

# --- Main Logic ---

echo "🚀 Starting continuous sync script. Will sync every 30 minutes."
echo "   Press [CTRL+C] to stop the script."

# Infinite loop to run the sync command periodically.
while true
do
    echo ""
    echo "▶️  Starting sync at $(date)..."
    
    # Check if the source directory exists before attempting to sync
    if [ ! -d "$SOURCE_DIR" ]; then
        echo "⚠️  Warning: Source directory '$SOURCE_DIR' not found. Skipping this cycle."
    else
        # Execute the AWS S3 sync command.
        aws s3 sync "$SOURCE_DIR" "$S3_DESTINATION"
        echo "✅  Sync completed successfully."
    fi

    echo "💤  Sleeping for ${SYNC_INTERVAL_SECONDS} seconds (30 minutes) until the next run."
    sleep $SYNC_INTERVAL_SECONDS
done