import torch
from arguments import parser
from datasets import get_dataset_model
from attacker import AdversarialAttack
import config


def main():
    """
        Main function to run adversarial attacks on the trained model.
    """
    args = parser.parse_args()
    device = config.DEVICE

    # Load the best model
    model_path = f'{config.LOGGING_PATH}/best_model.pth'
    model = torch.load(model_path).to(device)

    # Get the data loader
    _, data_loader, _ = get_dataset_model()

    # Initialize adversarial attack
    adversarial_attack = AdversarialAttack(model, args.attack_method, args.attack_type)

    # Run the specified attack
    if args.attack_type == "targeted":
        print(f"Running targeted {args.attack_method} attack on class {args.target_class} with epsilon {args.epsilon}")
        adversarial_accuracy = adversarial_attack.run(data_loader, args.epsilon, args.target_class)
    else:
        print(f"Running untargeted {args.attack_method} attack with epsilon {args.epsilon}")
        adversarial_accuracy = adversarial_attack.run(data_loader, args.epsilon)

    # Print the attack accuracy
    print(f"Attack accuracy: {adversarial_accuracy}")


if __name__ == '__main__':
    main()
