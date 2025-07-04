#!/bin/bash

BASE_DIR="saved/models"
LOG_DIR="saved/log"

find "$BASE_DIR" -mindepth 2 -maxdepth 2 -type d | while read -r exec_dir; do
	if [ ! -f "$exec_dir/model_best.pth" ]; then
		echo "Removing $exec_dir"
		rm -rf "$exec_dir"

		# Extract model and execution names
		model_name=$(basename "$(dirname "$exec_dir")")
		exec_name=$(basename "$exec_dir")
		log_exec_dir="$LOG_DIR/$model_name/$exec_name"
		
		if [ -d "$log_exec_dir" ]; then
			echo "Removing $log_exec_dir"
			rm -rf "$log_exec_dir"
		fi
	fi
done