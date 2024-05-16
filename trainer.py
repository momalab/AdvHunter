import torch
from tqdm import tqdm
import config


class Trainer:
    """
        Trainer class to handle the training and evaluation of the model.
    """
    def __init__(self, model):
        """
            Initialize the Trainer with the model, criterion, and optimizer.
            Args:
                model (torch.nn.Module): The neural network model to train.
        """
        self.device = config.DEVICE
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM,
                                         weight_decay=config.WEIGHT_DECAY)

    def train(self, train_loader, test_loader):
        """
            Train the model using the training data loader and evaluate using the test data loader.
            Args:
                train_loader (DataLoader): The DataLoader for training data.
                test_loader (DataLoader): The DataLoader for testing data.
        """
        best_accuracy = 0.0
        patience_counter = 0

        for epoch in range(config.NUM_EPOCHS):
            self.model.train()
            running_loss = 0.0
            correct_train, total_train = 0, 0

            # Training loop
            progress_bar = tqdm(train_loader, desc=f'Training {epoch + 1}/{config.NUM_EPOCHS}', leave=False)
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            test_accuracy = self.evaluate(test_loader)

            print(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}]")
            print(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Test Accuracy: {test_accuracy:.4f}")

            # Save the best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(self.model, f"{config.LOGGING_PATH}/best_model.pth")
                print(f"Best model saved with accuracy: {best_accuracy:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered.")
                break

    def evaluate(self, data_loader):
        """
            Evaluate the model using the provided data loader.
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

    def load_model(self, path):
        """
            Load a model from the specified path.
            Args:
                path (str): The path to the saved model.
        """
        self.model = torch.load(path)
        self.model.to(self.device)
