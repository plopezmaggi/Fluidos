#!/bin/bash

# Set the working directory
working_directory="./"

# Loop through all subdirectories
for dir in "$working_directory"*/; do
  # Check if the directory contains any .mp4 files
  if [ -n "$(find "$dir" -maxdepth 1 -type f -name "*.mp4")" ]; then
    echo "Processing directory: $dir"
    
    # Loop through .mp4 files in the directory and rename them
    for mp4_file in "$dir"*.mp4; do
      # Extract the filename (without extension) from the path
      filename=$(basename "$mp4_file" .mp4)
      
      # Define the new filename (modify this as needed)
      new_filename="video.mp4"
      
      # Rename the .mp4 file
      mv "$mp4_file" "$dir$new_filename"
      
      echo "Renamed: $mp4_file -> $new_filename"
    done
  fi
done
