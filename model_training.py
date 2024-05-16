import os
from trainer import Trainer
from datasets import get_dataset_model
import config


def main():
    """
        Main function to train and evaluate the model.
    """
    # Ensure logging directory exists
    os.makedirs(config.LOGGING_PATH, exist_ok=True)

    # Get data loaders and model
    train_loader, test_loader, model = get_dataset_model()

    # Initialize the trainer
    trainer = Trainer(model)

    # Train the model
    trainer.train(train_loader, test_loader)

    # Load the best model
    trainer.load_model(f'{config.LOGGING_PATH}/best_model.pth')

    # Evaluate the model
    accuracy = trainer.evaluate(test_loader)
    print(f"Final Test Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
