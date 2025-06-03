#!/bin/bash

if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <train_val_dir> <val_ratio>"
	exit 1
fi

# get all class subdirectories within train_val directory

VAL_RATIO=$2
TRAIN_DIR="$1/train"
VAL_DIR="$1/val"

mkdir -p "$TRAIN_DIR"
mkdir -p "$VAL_DIR"

for class_dir in "$1"/train_val/*/; do
	class_name=$(basename "$class_dir")
	mkdir -p "$TRAIN_DIR/$class_name"
	mkdir -p "$VAL_DIR/$class_name"

	files=("$class_dir"/*.png)
	total=${#files[@]}
	val_count=$(awk "BEGIN {printf \"%d\", $total * $VAL_RATIO}")

	printf "Processing class '%s': %d files, %d for validation, %d for training\n" \
		"$class_name" "$total" "$val_count" "$((total - val_count))"

	# shuffle files
	shuffled=($(printf "%s\n" "${files[@]}" | shuf))

	# split into val and train
	for i in "${!shuffled[@]}"; do
		file="${shuffled[$i]}"
		if (( i < val_count )); then
			cp "$file" "$VAL_DIR/$class_name/"
		else
			cp "$file" "$TRAIN_DIR/$class_name/"
		fi
	done

done

echo "Training and validation sets created in $TRAIN_DIR and $VAL_DIR respectively."
