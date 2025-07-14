#!/bin/bash

# Create the files train.txt, val.txt, and test.txt in the specified directory
# for JacobMedDataset

if [[ $# -ne 3 ]]; then
	echo "Usage: $0 <dataset_dir> <val_ratio> <reduction_ratio>"
	echo "  reduction_ratio: fraction of original dataset to use (0.0-1.0)"
	exit 1
fi

MIN_SAMPLES=400

DATASET_DIR="$1"
VAL_RATIO=$2
REDUCTION_RATIO=$3

TRAIN_FILE="$DATASET_DIR/train.txt"
VAL_FILE="$DATASET_DIR/val.txt"
TEST_FILE="$DATASET_DIR/test.txt"

if (( $(awk "BEGIN {print ($REDUCTION_RATIO < 1.0)}") )); then
	TRAIN_FILE="$DATASET_DIR/train_reduced.txt"
	VAL_FILE="$DATASET_DIR/val_reduced.txt"
	TEST_FILE="$DATASET_DIR/test_reduced.txt"
fi

train_count=0
val_test_count=0

# Create the train.txt, val.txt, test.txt files
# Ensure the output files are empty before writing
> "$TRAIN_FILE"
> "$VAL_FILE"
> "$TEST_FILE"

# Create the train.txt and val.txt files

for class_dir in "$DATASET_DIR"/train/*; do
	class_name=$(basename "$class_dir")

	files=("$class_dir"/*.png)
	total=${#files[@]}
	
	# Apply reduction ratio to the total number of files
	reduced_total=$(awk "BEGIN {printf \"%d\", $total * $REDUCTION_RATIO}")
	
	# If reduced total is less than threshold, keep the lowest between threshold and total
	if (( reduced_total < MIN_SAMPLES )); then
		if (( total < MIN_SAMPLES )); then
			effective_total=$total
			printf "Warning: Class '%s' has only %d files (below threshold of %d) - using all files\n" \
				"$class_name" "$total" "$MIN_SAMPLES"
		else
			effective_total=$MIN_SAMPLES
			printf "Class '%s': %d total files, reduced to %d would be below threshold, using %d files\n" \
				"$class_name" "$total" "$reduced_total" "$MIN_SAMPLES"
		fi
	else
		effective_total=$reduced_total
	fi
	
	val_test_count=$(awk "BEGIN {printf \"%d\", $effective_total * $VAL_RATIO}")
	train_count=$((effective_total - val_test_count))

	printf "Processing class '%s': %d total files, %d after reduction (%.1f%%), %d for validation, %d for training\n" \
		"$class_name" "$total" "$effective_total" "$(awk "BEGIN {printf \"%.1f\", $REDUCTION_RATIO * 100}")" "$val_test_count" "$train_count"

	# shuffle files
	shuffled=($(printf "%s\n" "${files[@]}" | shuf))

	# split into val and train (only use first effective_total files)
	for i in "${!shuffled[@]}"; do
		if (( i >= effective_total )); then
			break
		fi
		file="${shuffled[$i]}"
		if (( i < val_test_count )); then
			echo "train/$class_name/$(basename "$file")" >> "$VAL_FILE"
		else
			echo "train/$class_name/$(basename "$file")" >> "$TRAIN_FILE"
		fi
	done
done

# Create the test.txt file

for class_dir in "$DATASET_DIR"/val/*; do
	class_name=$(basename "$class_dir")

	files=("$class_dir"/*.png)
	total=${#files[@]}
	
	# Apply reduction ratio to test set
	if (( $(awk "BEGIN {print ($REDUCTION_RATIO < 1.0)}") )); then
		reduced_total=$(awk "BEGIN {printf \"%d\", $total * $REDUCTION_RATIO}")
	else
		reduced_total=$total
	fi
	
	# If reduced total is less than threshold, keep the lowest between threshold and total
	if (( reduced_total < MIN_SAMPLES )); then
		if (( total < MIN_SAMPLES )); then
			effective_total=$total
			printf "Processing class '%s': %d files (below threshold) - using all files for testing\n" \
				"$class_name" "$total"
		else
			effective_total=$MIN_SAMPLES
			printf "Processing class '%s': %d total files, reduced to %d would be below threshold, using %d files for testing\n" \
				"$class_name" "$total" "$reduced_total" "$MIN_SAMPLES"
		fi
	else
		effective_total=$reduced_total
		printf "Processing class '%s': %d total files, %d after reduction (%.1f%%) for testing\n" \
			"$class_name" "$total" "$reduced_total" "$(awk "BEGIN {printf \"%.1f\", $REDUCTION_RATIO * 100}")"
	fi

	# shuffle files and take only effective_total
	shuffled=($(printf "%s\n" "${files[@]}" | shuf))
	
	for i in "${!shuffled[@]}"; do
		if (( i >= effective_total )); then
			break
		fi
		file="${shuffled[$i]}"
		echo "val/$class_name/$(basename "$file")" >> "$TEST_FILE"
	done
done