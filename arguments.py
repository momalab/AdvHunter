import argparse


# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--attack_type", type=str, choices=["untargeted", "targeted"], default="untargeted",
                    help="Type of attack: 'untargeted' (default) or 'targeted'.")
parser.add_argument("--attack_method", type=str, choices=["fgsm", "pgd", "deepfool"], default="fgsm",
                    help="Attack method: 'fgsm' (default), 'pgd', or 'deepfool'.")
parser.add_argument("--epsilon", type=float, default=8/255,
                    help="Perturbation magnitude (default: 8/255).")
parser.add_argument("--target_class", type=int, default=0,
                    help="Target class for targeted attacks (default: 0).")
parser.add_argument("--image_class", type=int, default=0,
                    help="Class of the image to be used during inference (default: 0).")
parser.add_argument("--image_index", type=int, default=0,
                    help="Index of the image to be used during inference (default: 0).")
parser.add_argument('--cache', action='store_true', help='Use cache performance counter data if available.')
