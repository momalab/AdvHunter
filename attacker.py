import torch
from torch.utils.data import DataLoader, TensorDataset
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from deepfool import deepfool
import os
import config
import numpy as np


class AdversarialAttack:
    """
        Class to handle the creation and evaluation of adversarial attacks.
    """
    def __init__(self, model, attack_method, attack_type):
        """
            Initialize the AdversarialAttack class.
            Args:
                model (torch.nn.Module): The trained model to attack.
                attack_method (str): The method of attack ('fgsm', 'pgd', 'deepfool').
                attack_type (str): The type of attack ('targeted' or 'untargeted').
        """
        self.model = model
        self.attack_method = attack_method
        self.attack_type = attack_type
        self.device = config.DEVICE
        self.benign_saved = os.path.exists(f"{config.LOGGING_PATH}/benign")

    def evaluate(self, data_loader):
        """
            Evaluate the model on the provided data loader.
            Args:
                data_loader (DataLoader): The DataLoader for evaluation data.
            Returns:
                float: The accuracy of the model on the evaluation data.
        """
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def create_attack(self, images, epsilon, target=None):
        """
            Create adversarial examples using the specified attack method.
            Args:
                images (torch.Tensor): Batch of input images.
                epsilon (float): Perturbation magnitude.
                target (torch.Tensor, optional): Target labels for targeted attacks.
            Returns:
                torch.Tensor: Adversarial examples.
        """
        # Choose the attack method
        if self.attack_method == "fgsm":
            return fast_gradient_method(self.model, images, epsilon, norm=config.NORM, y=target,
                                        targeted=target is not None)
        elif self.attack_method == "pgd":
            nb_iter = config.ATTACK_ITER
            eps_iter = epsilon / nb_iter
            return projected_gradient_descent(self.model, images, epsilon, eps_iter, nb_iter, norm=config.NORM,
                                              y=target, targeted=target is not None)
        else:
            return deepfool(self.model, images, overshoot=epsilon, y=target, targeted=target is not None)

    def run(self, data_loader, epsilon, target_class=None):
        """
            Run the adversarial attack on the data loader.
            Args:
                data_loader (DataLoader): The DataLoader for input data.
                epsilon (float): Perturbation magnitude.
                target_class (int, optional): Target class for targeted attacks.
            Returns:
                float: The accuracy of the model on the adversarial examples.
        """
        # Initialize counters and lists to store adversarial images and labels
        counter = np.zeros(config.NUM_CLASS, dtype=int)
        adversarial_images = []
        adversarial_labels = []

        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            if target_class is not None:
                # Create target labels for targeted attacks
                target_labels = torch.full_like(labels, target_class)
            else:
                target_labels = labels

            # Generate adversarial examples
            adversarial = self.create_attack(images, epsilon, target_labels if self.attack_type == "targeted" else None)

            # Save the benign and adversarial images
            self.save_images(images, adversarial, labels, epsilon, counter)

            adversarial_images.append(adversarial)
            adversarial_labels.append(target_labels)

        # Create a new DataLoader for the adversarial examples
        adversarial_dataset = TensorDataset(torch.cat(adversarial_images), torch.cat(adversarial_labels))
        adversarial_loader = DataLoader(adversarial_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Evaluate the model on the adversarial examples
        adversarial_accuracy = self.evaluate(adversarial_loader)
        return adversarial_accuracy

    def save_images(self, benign_images, adversarial_images, labels, epsilon, counter):
        """
            Save the benign and adversarial images along with their labels.
            Args:
                benign_images (torch.Tensor): Batch of benign images.
                adversarial_images (torch.Tensor): Batch of adversarial images.
                labels (torch.Tensor): True labels of the images.
                epsilon (float): Perturbation magnitude.
                counter (numpy.ndarray): Counter for each class to index the saved images.
        """
        # Define the directory to save the images
        directory_name = f"{config.LOGGING_PATH}/{self.attack_type}/{self.attack_method}_{epsilon}"
        os.makedirs(directory_name, exist_ok=True)

        # Open log files to record labels
        benign_log = open(f"{config.LOGGING_PATH}/benign_labels.log", 'a')
        adversarial_log = open(f"{config.LOGGING_PATH}/{self.attack_type}/{self.attack_method}_{epsilon}_labels.log", 'a')

        self.model.eval()
        with torch.no_grad():
            # Get predictions for benign and adversarial images
            benign_outputs = self.model(benign_images)
            _, benign_predicted = benign_outputs.max(1)
            adversarial_outputs = self.model(adversarial_images)
            _, adversarial_predicted = adversarial_outputs.max(1)

        for i in range(benign_images.size(0)):
            # Define the filename for saving the image
            image_filename = f"image_class_{labels[i].item()}_{counter[labels[i].item()]}.pth"

            # Save benign images if not already saved
            if not self.benign_saved:
                os.makedirs(f"{config.LOGGING_PATH}/benign", exist_ok=True)
                torch.save(benign_images[i].cpu().unsqueeze(0), f"{config.LOGGING_PATH}/benign/{image_filename}")
                benign_log.write(f"{benign_predicted[i].item()}\n")

            # Save adversarial images
            torch.save(adversarial_images[i].cpu().unsqueeze(0), f"{directory_name}/{image_filename}")
            adversarial_log.write(f"{adversarial_predicted[i].item()}\n")

            # Update counter for the current class
            counter[labels[i].item()] += 1

        # Close the log files
        benign_log.close()
        adversarial_log.close()
