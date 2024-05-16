import torch

import config
from arguments import parser

# Parse command-line arguments
args = parser.parse_args()

# Load the model
model = torch.load(f"{config.LOGGING_PATH}/best_model.pth", map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# Load the image
image = torch.load(f"{config.LOGGING_PATH}/{args.attack_type}/{args.attack_method}_{args.epsilon}/image_class_{args.image_class}_{args.image_index}.pth")

# Perform the forward pass
with torch.no_grad():  # Disable gradient calculation
    output = model(image)
