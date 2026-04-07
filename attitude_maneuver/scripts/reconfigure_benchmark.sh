#!/bin/bash

set -euo pipefail

# ====================== Configuration (Adjust paths if needed) ======================

ANNOTATIONS_DIR="benchmark/annotations"
CONSTELL_BASE="benchmark/constellations"
NEW_CONSTELL_BASE="benchmark/new_constellations"
WORK_DIR_BASE="./work_dir/reconfigure"
SPLITS=("test" "val_seen" "val_unseen")
# ===================================================================================

# Check if jq (JSON parser) is installed
if ! command -v jq &> /dev/null; then
    echo -e "\033[31mERROR: jq is not installed (required for JSON parsing)\033[0m"
    echo "Install command: Ubuntu/Debian: sudo apt install jq | Mac: brew install jq"
    exit 1
fi

# Iterate over each split (test/val_seen/val_unseen)
for split in "${SPLITS[@]}"; do
    echo -e "\n\033[32m=====================================\033[0m"
    echo -e "\033[32mProcessing split: $split\033[0m"
    echo -e "\033[32m=====================================\033[0m"

    # Path to the current split's annotation file
    anno_file="${ANNOTATIONS_DIR}/${split}.json"

    # Skip if annotation file does not exist
    if [ ! -f "$anno_file" ]; then
        echo -e "\033[33mWARNING: File not found $anno_file, skipping this split\033[0m"
        continue
    fi

    # Extract ID list from the JSON "ids" array using jq
    echo "Extracting ID list from $anno_file..."
    ids=($(jq -r '.ids[]' "$anno_file"))

    # Skip if ID list is empty
    if [ ${#ids[@]} -eq 0 ]; then
        echo -e "\033[33mWARNING: Empty 'ids' field in $anno_file, skipping\033[0m"
        continue
    fi
    echo "Successfully extracted ${#ids[@]} IDs, starting processing..."

    # Iterate over each ID in the list
    for id in "${ids[@]}"; do
        echo -e "\n--- Processing ID: $id ---"

        # Calculate subdirectory: id // 1000 (integer division), formatted to 2 digits with leading zero
        subdir=$((id / 1000))
        subdir_02=$(printf "%02d" "$subdir")
        # Format ID to 5 digits with leading zeros
        id_05=$(printf "%05d" "$id")

        # Build full file paths
        constellation_path="${CONSTELL_BASE}/${split}/${subdir_02}/${id_05}.json"
        save_path="${NEW_CONSTELL_BASE}/${split}/${subdir_02}/${id_05}.json"
        work_dir="${WORK_DIR_BASE}/${split}/${id}"

        if [ -f "$save_path" ]; then
            echo -e "\033[33m  SKIP: Output file already exists → $save_path\033[0m"
            continue
        fi

        # Skip if original constellation file does not exist
        if [ ! -f "$constellation_path" ]; then
            echo -e "\033[33mSKIP: Original file not found $constellation_path\033[0m"
            continue
        fi

        # Auto-create output directories (recursive)
        mkdir -p "$(dirname "$save_path")" "$work_dir"

        # Execute the core Python command
        echo "Running command: python -m attitude_maneuver.scripts.run_lmrp_tune ..."
        python -m attitude_maneuver.scripts.run_lmrp_tune attitude_maneuver/config_lmrp.py \
            --work-dir "$work_dir" \
            --constellation "$constellation_path" \
            --save-path "$save_path"

        echo -e "\033[32mID $id processed successfully\033[0m"
    done

    echo -e "\n\033[32mAll IDs for split $split completed!\033[0m"
done

echo -e "\n\033[32m=====================================\033[0m"
echo -e "\033[32m✅ All tasks finished successfully!\033[0m"
echo -e "\033[32m=====================================\033[0m"
