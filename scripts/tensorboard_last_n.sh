#!/bin/bash
# This script runs TensorBoard on the last 5 training runs in a specified folder
# for each model.

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <root_dir> <num_execs>"
  exit 1
fi

ROOT_DIR=$1
NUM_EXECS=$2
TMP_DIR="tmp/tensorboard_visualization"

printf "Starting TensorBoard for the last %s executions in %s\n" "$NUM_EXECS" "$ROOT_DIR"

# get last N executions for each subfolder (each subfolder has folders which are the runs)
last_execs=$(find "$ROOT_DIR" -mindepth 2 -maxdepth 2 -type d | \
	awk -F/ '{print $(NF-1) "/" $NF}' | \
	sort -t/ -k1,1 -k2,2r | \
	awk -F/ '{
		count[$1]++
		if (count[$1] <= NUM_EXECS) print "'"$ROOT_DIR"'/"$1"/"$2
	}' NUM_EXECS="$NUM_EXECS")

printf "Found %d runs:\n" "$(echo "$last_execs" | wc -l)"

echo "$last_execs" | while read -r run; do
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

echo "$last_execs" | while read -r run; do
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
