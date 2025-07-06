#!/bin/bash
# This script runs TensorBoard on the last N training runs in specified folders
# for each model.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <num_execs> <root_dir1> [root_dir2] [root_dir3] ..."
  echo "  num_execs: Number of recent executions to include per model"
  echo "  root_dir*: One or more root directories containing training runs"
  exit 1
fi

NUM_EXECS=$1
shift  # Remove first argument, remaining args are root directories
ROOT_DIRS=("$@")
ROOT_DIRS=("$@")
TMP_DIR="tmp/tensorboard_visualization"

printf "Starting TensorBoard for the last %s executions in %d directories\n" "$NUM_EXECS" "${#ROOT_DIRS[@]}"

# Process each root directory
all_execs=""
for ROOT_DIR in "${ROOT_DIRS[@]}"; do
  if [[ ! -d "$ROOT_DIR" ]]; then
    echo "Warning: Directory $ROOT_DIR does not exist, skipping..."
    continue
  fi
  
  printf "Processing directory: %s\n" "$ROOT_DIR"
  
  # get last N executions for each subfolder (each subfolder has folders which are the runs)
  last_execs=$(find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -type d | \
    awk -F/ '{print $(NF-1) "/" $NF}' | \
    sort -t/ -k1,1 -k2,2r | \
    awk -F/ '{
      count[$1]++
      if (count[$1] <= NUM_EXECS) print "'"$ROOT_DIR"'/"$2
    }' NUM_EXECS="$NUM_EXECS")
  
  # Append to all_execs
  if [[ -n "$last_execs" ]]; then
    if [[ -z "$all_execs" ]]; then
      all_execs="$last_execs"
    else
      all_execs="$all_execs"$'\n'"$last_execs"
    fi
  fi
done

printf "Found %d total runs:\n" "$(echo "$all_execs" | wc -l)"

echo "$all_execs" | while read -r run; do
  config_file="${run/log/models}/config.json"
  log_file="$run/info.log"
  echo " - $run"
  echo "         Config: $config_file"
  echo "         Log: $log_file"
done

# create a tmp folder to store the symbolic links to these files
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

if [[ $? -ne 0 ]]; then
  echo "Error creating temporary directory $TMP_DIR"
  exit 1
fi

echo "Created temporary directory $TMP_DIR"

echo "$all_execs" | while read -r run; do
	parent_dir="$(basename "$(dirname "$run")")"
	run_dir="$(basename "$run")"
	target_dir="$TMP_DIR/$parent_dir"
	mkdir -p "$target_dir"
	ln -s "$(realpath "$run")" "$target_dir/$run_dir"
done

printf "Running TensorBoard in %s\n" "$TMP_DIR"
tensorboard --logdir "$TMP_DIR"

# remove tmp folder
rm -rf "$TMP_DIR"
echo "Temporary directory $TMP_DIR removed"
