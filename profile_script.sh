#!/bin/bash

# Function to process images for benign or adversarial cases
process_images() {
    local MODE=$1  # Mode of processing: benign or adversarial
    local IMAGE_CLASS  # Image class (0 to 9)
    local INDEX  # Image index (0 to 999)
    PYTHON_PATH=$(which python)  # Path to the Python interpreter

    # Loop through all image classes (0 to 9)
    for IMAGE_CLASS in {0..9}; do
      # Loop through all image indices (0 to 999)
        for INDEX in {0..999}; do
            if [ "$MODE" == "benign" ]; then
                echo "Processing benign images for class $IMAGE_CLASS, index $INDEX"
                # Run performance profiling for benign image processing
                sudo perf stat -C 0 -e "$PERF_EVENTS" -r 5 -o logs/perf_benign"$LOG_FILE_SUFFIX" --append \
                    taskset -c 0 "$PYTHON_PATH" prediction_benign.py --image_class="$IMAGE_CLASS" --image_index="$INDEX"
            else
                # Define log file name for adversarial attack
                local LOG_FILE_NAME="perf_${ATTACK_TYPE}_${ATTACK_METHOD}_${EPSILON}${LOG_FILE_SUFFIX}"
                echo "Processing $ATTACK_TYPE attack using $ATTACK_METHOD method (epsilon=$EPSILON) for class $IMAGE_CLASS, index $INDEX"
                # Run performance profiling for adversarial image processing
                sudo perf stat -C 0 -e "$PERF_EVENTS" -r 5 -o logs/"$LOG_FILE_NAME" --append \
                    taskset -c 0 "$PYTHON_PATH" prediction_adversarial.py --attack_type="$ATTACK_TYPE" --attack_method="$ATTACK_METHOD" --eps="$EPSILON" --image_class="$IMAGE_CLASS" --image_index="$INDEX"
            fi
        done
    done
}

# Main function to determine the mode and settings for processing images
main() {
    LOG_FILE_SUFFIX=".log"  # Default log file suffix
    PERF_EVENTS="branches,branch-misses,cache-references,cache-misses,instructions"
    if [ $# -eq 0 ]; then
        # If no arguments, process benign images with default settings
        process_images "benign"
    elif [ $# -eq 1 ]; then
        # If one argument, process benign images with cache-specific settings
        LOG_FILE_SUFFIX="_cache.log"
        PERF_EVENTS="L1-dcache-load-misses,L1-icache-load-misses,LLC-load-misses,LLC-store-misses"
        process_images "benign"
    else
        # If multiple arguments, process adversarial images with provided settings
        ATTACK_TYPE=$1  # Type of adversarial attack
        ATTACK_METHOD=$2  # Method of adversarial attack
        EPSILON=$3  # Epsilon value for the attack
        IS_CACHE_ENABLED=$4  # Cache profiling flag
        if [ "$IS_CACHE_ENABLED" == "cache" ]; then
            # Set cache-specific log file suffix and performance events
            LOG_FILE_SUFFIX="_cache.log"
            PERF_EVENTS="L1-dcache-load-misses,L1-icache-load-misses,LLC-load-misses,LLC-store-misses"
        else
            # Set default log file suffix and performance events
            LOG_FILE_SUFFIX=".log"
            PERF_EVENTS="branches,branch-misses,cache-references,cache-misses,instructions"
        fi
        process_images "adversarial"
    fi
}

# Run the main function with all passed arguments
main "$@"
