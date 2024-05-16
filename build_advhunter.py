import json
import numpy as np
from sklearn.mixture import GaussianMixture
from arguments import parser
import config
import os


def load_data(filepath):
    """
        Load data from a JSON file.
        Args:
            filepath (str): Path to the JSON file.
        Returns:
            dict: Data loaded from the JSON file.
    """
    if not os.path.exists(filepath):
        print(f'{filepath} does not exist.')
        return None

    # Open and read the JSON file
    with open(filepath, 'r') as file:
        return json.load(file)


def remove_outliers(data):
    """
        Remove outliers from the data based on mean and standard deviation.
        Args:
            data (array-like): Input data.
        Returns:
            list: Data with outliers removed.
    """
    # Calculate the mean and standard deviation of the data
    mean, std_dev = np.mean(data), np.std(data)

    # Define lower and upper bounds for outliers
    lower_bound = mean - config.STD_DEV_MULTIPLIER * std_dev
    upper_bound = mean + config.STD_DEV_MULTIPLIER * std_dev

    # Filter out the outliers
    return [x for x in data if lower_bound <= x <= upper_bound]


def find_thresholds(data, gmm):
    """
        Find thresholds for anomaly detection using Gaussian Mixture Model (GMM).
        Args:
            data (array-like): Input data.
            gmm (GaussianMixture): Trained GMM.
        Returns:
            float: Threshold value for anomaly detection.
    """
    # Calculate the negative log likelihood scores for the data
    scores = [-gmm.score_samples(sample.reshape(-1, 1)) for sample in data]

    # Calculate mean and standard deviation of the scores
    mean, std_dev = np.mean(scores), np.std(scores)

    # Define the threshold for anomaly detection
    return mean + config.THRESHOLD_MULTIPLIER * std_dev


def find_best_gmm(data):
    """
        Find the best GMM for each class in the data.
        Args:
            data (dict): Data split by class.
        Returns:
            tuple: List of best GMMs and their corresponding thresholds.
    """
    best_gmms, thresholds = [], []

    for c in range(config.NUM_CLASS):
        # Sample data for the current class
        sampled_data = np.random.choice(data[c], config.NUM_SAMPLES, replace=False)
        best_gmm, lowest_bic = None, np.infty

        # Iterate over possible number of components (peaks) in GMM
        for n_peaks in range(1, config.MAX_PEAKS + 1):
            # Initialize and fit the GMM
            gmm = GaussianMixture(n_components=n_peaks, max_iter=config.GMM_MAX_ITER, n_init=config.GMM_N_INIT)
            gmm.fit(sampled_data.reshape(-1, 1))

            # Calculate Bayesian Information Criterion (BIC) for the model
            bic = gmm.bic(sampled_data.reshape(-1, 1))

            # Select the model with the lowest BIC
            if bic < lowest_bic:
                lowest_bic, best_gmm = bic, gmm

        # Find the threshold for the best GMM
        thresholds.append(find_thresholds(sampled_data, best_gmm))

        # Store the best GMM for the current class
        best_gmms.append(best_gmm)

    return best_gmms, thresholds


def split_data(data, labels):
    """
        Split data into categories based on labels and remove outliers.
        Args:
            data (array-like): Input data.
            labels (array-like): Corresponding labels.
        Returns:
            dict: Categorized data with outliers removed.
    """
    # Initialize a dictionary to store categorized data
    categorized_data = {label: [] for label in np.unique(labels)}

    # Categorize the data based on labels
    for d, l in zip(data, labels):
        categorized_data[l].append(d)

    # Remove outliers from each category
    return {label: remove_outliers(np.array(values)) for label, values in categorized_data.items()}


def analyze_event(event, benign_data, adversarial_data, benign_labels, adversarial_labels):
    """
        Analyze a specific event by comparing benign and adversarial data.
        Args:
            event (str): The event to analyze.
            benign_data (dict): Benign data.
            adversarial_data (dict): Adversarial data.
            benign_labels (array-like): Labels for benign data.
            adversarial_labels (array-like): Labels for adversarial data.
    """
    print(f"---------------------\nEvent: {event}\n=====================")

    # Split benign and adversarial data by class
    benign_split = split_data(benign_data[event], benign_labels)
    adversarial_split = split_data(adversarial_data[event], adversarial_labels)

    # Find the best GMMs and thresholds for benign data
    gmms, thresholds = find_best_gmm(benign_split)

    overall_acc, overall_f_score = 0, 0

    for c in range(config.NUM_CLASS):
        tp, fp, tn, fn = 0, 0, 0, 0

        # Evaluate benign samples
        for sample in benign_split[c]:
            score = -gmms[c].score_samples(sample.reshape(-1, 1))
            if score > thresholds[c]:
                fp += 1  # False positive
            else:
                tn += 1  # True negative

        # Evaluate adversarial samples
        for sample in adversarial_split[c]:
            score = -gmms[c].score_samples(sample.reshape(-1, 1))
            if score > thresholds[c]:
                tp += 1  # True positive
            else:
                fn += 1  # False negative

        # Calculate F1-score and accuracy for the current class
        f_score = (2 * tp) / (2 * tp + fp + fn)
        acc = (tp + tn) / (len(benign_split[c]) + len(adversarial_split[c]))
        overall_acc += acc
        overall_f_score += f_score

        print(f"Class: {c}, Accuracy: {np.round(acc * 100, 2)}, F1-score: {np.round(f_score, 4)}")

    # Calculate and print overall accuracy and F1-score
    print(f"Overall Accuracy: {np.round(overall_acc / config.NUM_CLASS * 100, 2)}, Overall F1-score: {np.round(overall_f_score / config.NUM_CLASS, 4)}")


def main():
    """
        Main function to load data, analyze events, and print results.
    """
    args = parser.parse_args()
    suffix = "_cache" if args.cache else ""

    # Load benign data and labels
    benign_data = load_data(f"{config.LOGGING_PATH}/perf_benign{suffix}.json")
    benign_labels = np.loadtxt(f"{config.LOGGING_PATH}/benign_labels.log")

    # Load adversarial data and labels
    adversarial_data = load_data(f"{config.LOGGING_PATH}/perf_{args.attack_type}_{args.attack_method}_{args.epsilon}{suffix}.json")
    adversarial_labels = np.loadtxt(f"{config.LOGGING_PATH}/{args.attack_type}/{args.attack_method}_{args.epsilon}.log")

    # Define events to analyze based on the cache flag
    events = ['branches', 'branch-misses', 'cache-references', 'cache-misses', 'instructions'] if not args.cache else \
             ['L1-dcache-load-misses', 'L1-icache-load-misses', 'LLC-load-misses', 'LLC-store-misses']

    # Ensure data is loaded before proceeding
    if benign_data and adversarial_data:
        print(f"Attack: {args.attack_method} {args.attack_type}, Epsilon: {args.epsilon}")

        # Analyze each event
        for event in events:
            analyze_event(event, benign_data, adversarial_data, benign_labels, adversarial_labels)


if __name__ == "__main__":
    main()
