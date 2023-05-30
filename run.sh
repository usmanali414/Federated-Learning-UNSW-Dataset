#!/bin/bash
# Define the directory to list
dir="./data/client_dataset"

# Create an empty array to store the file names
files=()

# Loop through each file in the directory
for file in $dir/*
do
  # Check if the file is a regular file and ends with ".csv"
  if [[ -f $file && $file == *.csv ]]; then
    # Add the file to the array
    files+=("$file")
  fi
done

# Print the array of file names
echo "${files[@]}"
length=${#files[@]}
last_index_for_array=$length
((last_index_for_array--))


python server.py &
sleep 5 # Sleep for 2s to give the server enough time to start

for i in `seq 0 1 ${last_index_for_array}`; do
    echo "Starting client $i"
    python client.py -p "${files[i]}" -c "${i}" &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
