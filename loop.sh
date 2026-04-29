#!/bin/bash
#
# Runs a command line 10 times and stops when an execution failed once.
# Logs from the failed process are stored in file.txt.
# 
# Usage: 
#   ```sh
#   bash loop.sh --max 10 --output file.txt torchrun --nproc_per_node 8 test_script.py
#   ``` 

# Default values for max runs and output file
MAX_RUNS=1
OUTPUT_FILE=""

# Parse options --max and --output
while [[ "$1" =~ ^- && ! "$1" == "--" ]]; do
  case $1 in
    --max )
      shift
      MAX_RUNS=$1
      ;;
    --output )
      shift
      OUTPUT_FILE=$1
      ;;
  esac
  shift
done
if [[ "$1" == '--' ]]; then shift; fi

# The command to run
COMMAND="$@"

# Run the command the specified number of times
for ((i=1; i<=MAX_RUNS; i++))
do
    echo "Run $i"
    $COMMAND
    
    # Check the exit status of the last command
    if [ $? -ne 0 ]; then
        echo "Command failed on run $i, logging to $OUTPUT_FILE."
        # Redirect both stdout and stderr to the output file (overwrite the file)
        $COMMAND > "$OUTPUT_FILE" 2>&1
        break
    fi
done
