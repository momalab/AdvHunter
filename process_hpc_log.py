import json
import os

import config
from arguments import parser


def read_log_file(filepath, is_cache):
    """
        Read and parse a log file to extract hardware performance counters (HPC) data.
        Args:
            filepath (str): Path to the log file.
            is_cache (bool): Flag to indicate if the log file is cache-related.
        Returns:
            dict: Parsed HPC data.
    """
    if not os.path.exists(filepath):
        print(f'{filepath} does not exist.')
        return None

    # Initialize HPC data structure
    hpc_data = {
        'branches': [],
        'branch-misses': [],
        'cache-references': [],
        'cache-misses': [],
        'instructions': []
    } if not is_cache else {
        'L1-dcache-load-misses': [],
        'L1-icache-load-misses': [],
        'LLC-load-misses': [],
        'LLC-store-misses': [],
    }

    # Read and parse the log file
    with open(filepath, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            # Skip header lines and other irrelevant lines
            if not tokens or tokens[0] in {"#", "Performance"} or tokens[3] == "seconds":
                continue
            # Extract the relevant HPC data
            hpc_data[tokens[1]].append(float(tokens[0].replace(",", "")))

    return hpc_data


def write_to_json(hpc_data, filepath):
    """
        Write the HPC data to a JSON file.
        Args:
            hpc_data (dict): HPC data to write.
            filepath (str): Path to the output JSON file.
    """
    with open(filepath, 'w') as file:
        json.dump(hpc_data, file)


def process_log_file(log_type, attack_type=None, attack_method=None, epsilon=None, is_cache=False):
    """
        Process a log file and convert its data to a JSON file.
        Args:
            log_type (str): Type of the log file ('benign' or 'adversarial').
            attack_type (str, optional): Type of adversarial attack (required if log_type is 'adversarial').
            attack_method (str, optional): Method of adversarial attack (required if log_type is 'adversarial').
            epsilon (float, optional): Perturbation magnitude (required if log_type is 'adversarial').
            is_cache (bool): Flag to indicate if the log file is cache-related.
    """
    suffix = "_cache" if is_cache else ""

    if log_type == "benign":
        log_filepath = f"{config.LOGGING_PATH}/perf_benign{suffix}.log"
        json_filepath = f"{config.LOGGING_PATH}/perf_benign{suffix}.json"
    else:
        log_filepath = f"{config.LOGGING_PATH}/perf_{attack_type}_{attack_method}_{epsilon}{suffix}.log"
        json_filepath = f"{config.LOGGING_PATH}/perf_{attack_type}_{attack_method}_{epsilon}{suffix}.json"

    # Read HPC data from the log file
    hpc_data = read_log_file(log_filepath, is_cache)
    if hpc_data is not None:
        # Write HPC data to a JSON file
        write_to_json(hpc_data, json_filepath)


def main():
    """
        Main function to process log files for benign and adversarial examples.
    """
    args = parser.parse_args()

    # Process benign log file
    process_log_file(log_type="benign", is_cache=args.cache)

    # Process adversarial log file
    process_log_file(log_type="adversarial", attack_type=args.attack_type,
                     attack_method=args.attack_method, epsilon=args.epsilon, is_cache=args.cache)


if __name__ == "__main__":
    main()
