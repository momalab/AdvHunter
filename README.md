# AdvHunter: Detecting Adversarial Perturbations in Black-Box Neural Networks through Hardware Performance Counters

## 📑 Overview
AdvHunter is a defense framework designed to protect Deep Neural Networks (DNNs) from _adversarial examples_, even in black-box scenarios where the network's internal details are unknown. It leverages Hardware Performance Counters (HPCs) to monitor the microarchitectural activities of a DNN during inference. By applying Gaussian Mixture Models to the collected HPC data, AdvHunter identifies anomalies indicating whether an input is legitimate or has been altered by adversarial perturbations.

## 🖥️  System Requirements
---
- **Operating System**: Linux (Tested on Ubuntu 18.04.6 LTS).
- **Processor**: Intel (Tested on Intel i7-9700).
- **Python Version**: Python 3.10.9 (Confirmed compatibility).
- **CUDA Toolkit**: Optional for GPU acceleration during training (Tested with version 11.5, V11.5.119).

## 🛠️ Installation Guide

---
- Ensure the `perf` tool is installed on your system.
- Set up a dedicated Python virtual environment and install required dependencies:
   ```bash
   python -m venv advhunter
   source advhunter/bin/activate
   pip install -r requirements.txt
   ```

## 🚀 Step-by-Step Execution Guide

---
1. **Model Training**: Train a ResNet18 model on the CIFAR10 dataset.
   ```bash
    python model_training.py
   ```
   The best-performing model is saved in `logs/best_model.pth`.


2. **Adversarial Examples Generation**: Generate adversarial examples by specifying various arguments.
   ```bash
   python adversarial_examples.py --model=<model> --attack_type=<attack> --attack_method=<method> --epsilon=<epsilon> --target_class=<target>
   ```
   The supported arguments are:
   - **Attack Type** (`--attack_type`): Specify `targeted` or `untargeted`.
   - **Attack Method** (`--attack_method`): Specify `fgsm`, `pgd`, or `deepfool`.
   - **Perturbation Strength** (`--epsilon`): Set the perturbation strength.
   - **Target Class** (`--target_class`): For `targeted` attacks, specify the misclassification target class.

   Outputs:
   - **Benign Images**: Saved in `logs/benign` directory.
   - **Predicted Benign Labels**: Logged in `logs/benign_labels.log` file.
   - **Adversarial Images**: Saved in `logs/<attack_type>/<attack_method>_<epsilon>` directory.
   - **Predicted Adversarial Labels**: Logged in `logs/<attack_type>/<attack_method>_<epsilon>_labels.log` file.


3. **Profile Performance Counters**: Profile hardware performance counters using the `perf` tool during inference of both benign and adversarial image sets. Superuser access is required to run the `perf` tool. 

   - For Benign Images.
     ```bash
     ./profile_script.sh [cache]
     ```
   - For Adversarial Images.
     ```bash
     ./profile_script.sh <attack_type> <attack_method> <epsilon> [cache]
     ```
   Include the optional `cache` argument for both to collect cache-based performance counter data. Performance counter data for benign images is logged in `logs/perf_benign.log` and for adversarial images in `logs/perf_<attack_type>_<attack_method>_<epsilon>.log`.


4. **Process Performance Counters Data**: Convert the logged performance counter data into a structured JSON format. 
   ```bash
   python process_hpc_log.py --attack_type=<attack> --attack_method=<method> --epsilon=<epsilon> [--cache]
   ```
   Include the optional `--cache` argument to process performance counter data for cache-based events. The processed data files are saved in `logs/perf_benign.json` and `logs/perf_<attack_type>_<attack_method>_<epsilon>.json`.


5. **Anomaly Detection and Model Evaluation**: Construct Gaussian Mixture Models using performance counter data and predicted labels for benign images. Use these models and prediced labels for adversarial images to detect anomalies. The framework's detection capability is quantified using `accuracy` and `F1-score` metrics.
   ```bash
   python build_advhunter.py --attack_type=<attack> --attack_method=<method> --epsilon=<epsilon> [--cache]
   ```
   Include the optional `--cache` argument to analyze cache-based performance counter data.


6. **Reproducibility**: In order to reproduce the results presented in the paper, following files are included in the `reproducibility` directory.
    - `best_model.pth`
    - `benign_labels.log`
    - `untargeted/fgsm_0.1_labels.log`
    - `perf_benign.log`
    - `perf_untargeted_fgsm_0.1.log`

## 📚 Cite Us

---
If you find our work interesting and use it in your research, please cite our paper describing:

Manaar Alam and Michail Maniatakos, "_AdvHunter: Detecting Adversarial Perturbations in Black-Box Neural Networks through Hardware Performance Counters_", DAC 2024.

### BibTex Citation
```
@inproceedings{DBLP:conf/dac/AlamM24,
  author       = {Manaar Alam and
                  Michail Maniatakos},
  title        = {{AdvHunter: Detecting Adversarial Perturbations in Black-Box Neural Networks through Hardware Performance Counters}},
  booktitle    = {61st {ACM/IEEE} Design Automation Conference, {DAC} 2024, San Francisco,
                  CA, USA, June 23-27, 2024},
  publisher    = {{IEEE}},
  year         = {2024}
}
```

---

## 📩 Contact Us

---
For more information or help with the setup, please contact Manaar Alam at: [alam.manaar@nyu.edu](mailto:alam.manaar@nyu.edu)