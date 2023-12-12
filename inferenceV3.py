import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import matplotlib.pyplot as plt
import time

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your trained U-Net model
model = UNET(in_channels=3, out_channels=1).to(device)
checkpoint = torch.load('my_checkpoint.pth.tar')
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.eval()

# Specify input and output directories
input_dir = 'data/test_images'
output_dir = 'data/output_images'
overlay_output_dir = 'data/overlay_images'  # Add a new directory for overlaid images

# Create the output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(overlay_output_dir, exist_ok=True)

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((950, 600)),  # Resize to match the model's input size
    transforms.ToTensor(),
    # Add any other necessary transformations
])

# Create a custom dataset for the test images
class Prediction(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load input image
        input_image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)

        return {'image': input_image, 'image_name': image_name}

dataset = Prediction(input_dir, transform=transform)

# Create a data loader for the test dataset
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Set threshold for binary segmentation
threshold = 0.5  # Adjust

# Set kernel for morphological operations
kernel_size = 7
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Initialize a list to store average widths
average_widths = []

# Measure the time taken for predictions
start_time = time.time()

# Iterate through each batch of images in the test dataset
for batch in data_loader:
    # Move the batch to the appropriate device
    images = batch['image'].to(device)

    # Make predictions using the U-Net model
    with torch.no_grad():
        pred_mask = model(images)

    # Probability Threshold
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.9).float()

    # Convert the predicted mask to a numpy array
    binary_mask_np = pred_mask.squeeze().cpu().numpy()

    # Convert to uint8
    binary_mask_np_uint8 = (binary_mask_np * 255).astype(np.uint8)

    # Apply morphological operations
    binary_mask_np = cv2.erode(binary_mask_np_uint8, kernel, iterations=1)
    binary_mask_np = cv2.dilate(binary_mask_np, np.ones((15, 15), np.uint8), iterations=1)

    # Save the result mask to the output directory
    image_name = batch['image_name'][0]
    output_path = os.path.join(output_dir, f'result_mask_{image_name}')
    cv2.imwrite(output_path, binary_mask_np_uint8)

    # Distance Transform to measure average width
    distance_transform = cv2.distanceTransform(binary_mask_np_uint8, cv2.DIST_L2, 5)

    # Calculate the average width
    average_width = np.mean(distance_transform[binary_mask_np_uint8 > 0])

    # Store the average width in the list
    average_widths.append(average_width)

    # Overlay the predicted mask on the input image
    original_image = images.squeeze().cpu().numpy().transpose(1, 2, 0) * 255  # Convert to 0-255 scale
    overlay = original_image.copy()  # Copy the input image
    overlay[binary_mask_np_uint8 > 0] = [0, 0, 255]  # Set overlay color where mask is 1 (assuming blue color)
    overlay = cv2.addWeighted(overlay.astype(np.uint8), 0.3, original_image.astype(np.uint8), 0.7, 0)

    # Save the overlaid image to the overlay output directory
    overlay_output_path = os.path.join(overlay_output_dir, f'overlay_{image_name}')
    cv2.imwrite(overlay_output_path, overlay)

    # Print or save the average width
    print(f"Average Width for {image_name}: {average_width}")

# Calculate and print the total time taken
total_time = time.time() - start_time
print(f"Total time taken for predictions: {total_time:.2f} seconds")

# Plot the average thicknesses
plt.plot(average_widths, marker='o', linestyle='-', color='b', markersize=3)
plt.xlabel('Image Index')
plt.ylabel('Average Width')
plt.title('Average Width Across Images')
# Set y-axis limits to a custom range, adjust these values based on your data
plt.ylim(5, 10)  # Change 0 and 20 to your desired limits
# Adjust the spacing between x-axis ticks
plt.xticks(np.arange(0, len(average_widths), step=10))
plt.show()
