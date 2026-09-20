import os
import random
import shutil

# Original dataset path
SOURCE_DIR = r"D:\xiazai\fruit"

# New dataset path
TARGET_DIR = r"D:\xiazai\fruit_dataset"

# Number of images to randomly select from each class
NUM_IMAGES = 100

# Set a random seed to make the selection reproducible
random.seed(42)

# Four classes
classes = ["apple", "banana", "Mango", "orange"]

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for class_name in classes:

    # Source and target folder paths
    source_folder = os.path.join(SOURCE_DIR, class_name)
    target_folder = os.path.join(TARGET_DIR, class_name)

    # Create the target folder if it does not exist
    os.makedirs(target_folder, exist_ok=True)

    # Get all image files from the source folder
    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith(image_extensions)
    ]

    print(f"{class_name}: Found {len(images)} images")

    # Check whether there are enough images
    if len(images) < NUM_IMAGES:
        print(f"❌ {class_name}: Not enough images (less than 100)!")
        continue

    # Randomly select 100 images
    selected_images = random.sample(images, NUM_IMAGES)

    # Copy the selected images to the target folder
    for image in selected_images:
        source_path = os.path.join(source_folder, image)
        target_path = os.path.join(target_folder, image)

        shutil.copy2(source_path, target_path)

    print(f"✅ {class_name}: Selected {NUM_IMAGES} images")


print("\nAll classes have been processed!")
print(f"New dataset location: {TARGET_DIR}")