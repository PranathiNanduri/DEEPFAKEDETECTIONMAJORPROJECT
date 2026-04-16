import os
import random
import shutil

SOURCE_DIR = "data/processed_faces"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

classes = ["real", "fake", "spoof"]

split_ratio = 0.8

for cls in classes:
    source_folder = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(source_folder)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(VAL_DIR, cls, img)
        )

print("Dataset split complete successfully!")