from datasets import load_dataset
from PIL import Image
import random

def preprocess_image(image):
    # Convert the image to RGB mode
    image = image.convert('RGB')
    
    # Resize the image to a larger size (e.g., 256x256) while maintaining the aspect ratio
    image.thumbnail((256, 256))
    
    # Perform random cropping
    crop_size = 224
    left = random.randint(0, image.width - crop_size)
    top = random.randint(0, image.height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

# Load the ImageNet dataset
test_set = load_dataset('imagenet-1k', split='train')

# Get an example image and label
image = test_set[1]['image']
label = test_set[1]['label']

# Preprocess the image
preprocessed_image = preprocess_image(image)

# Print the original image, preprocessed image, and label
print("Original Image:")
print(image)
print("\nPreprocessed Image:")
print(preprocessed_image)
print("\nLabel:")
print(label)
