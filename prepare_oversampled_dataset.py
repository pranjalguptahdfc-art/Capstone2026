"""
Prepare oversampled dataset for banana detector training.

This script:
1. Loads the Roboflow banana ripening dataset from detector_training_data
3. Randomly selects 286 unique training sources and 71 unique validation sources
4. Creates an oversampled training list with 5x repetition (286 * 5 = 1430 entries)
5. Creates a held-out validation list with 71 entries (no repetition)
6. Saves:
   - oversampled_image_paths.txt (the oversampled training list)
   - heldout_validation_paths.txt
   - data_oversampled.yaml (YAML config pointing to the above files)
7. All outputs saved to data/oversampled/ directory

The script ensures no overlap between training and validation sources.
"""

import os
import random
import shutil
from pathlib import Path
import yaml

# Set random seed for reproducibility
random.seed(42)

# Paths
project_dir = Path.cwd()
detector_training_data_dir = project_dir / "detector_training_data"
train_images_dir = detector_training_data_dir / "train" / "images"
valid_images_dir = detector_training_data_dir / "valid" / "images"
test_images_dir = detector_training_data_dir / "test" / "images"

# Output directory
output_dir = project_dir / "data" / "oversampled"
output_dir.mkdir(parents=True, exist_ok=True)

# Get all image paths in the training set
train_images = list(train_images_dir.glob("*.*"))
# Filter to only image files (common extensions)
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
train_images = [p for p in train_images if p.suffix.lower() in image_extensions]

print(f"Found {len(train_images)} images in training set.")

# We need to select 286 unique sources for oversampled training
# and 71 unique sources for held-out validation from the training set.
# We'll randomly sample without replacement.

# First, shuffle the list
random.shuffle(train_images)

# Select 286 for training sources
train_sources = train_images[:286]
# From the remaining, select 71 for validation sources
val_sources = train_images[286:286+71]

print(f"Selected {len(train_sources)} training sources and {len(train_sources)} training sources and {len(val_sources)} validation sources.")
print(f"Overlap check: {len(set(train_sources) & set(val_sources))} (should be 0)")

# Create oversampled training list: each source repeated 5 times
oversampled_training_list = []
for img_path in train_sources:
    # Repeat 5 times
    for _ in range(5):
        oversampled_training_list.append(str(img_path))

# Create held-out validation list (no repetition)
heldout_validation_list = [str(p) for p in val_sources]

# Save the oversampled image path list (which is the oversampled training list with repeats)
oversampled_image_paths_file = output_dir / "oversampled_image_paths.txt"
with open(oversampled_image_paths_file, "w") as f:
    f.write("\n".join(oversampled_training_list))
print(f"Saved oversampled image path list to {oversampled_image_paths_file} ({len(oversampled_training_list)} lines)")

# Save the held-out validation file list
heldout_validation_paths_file = output_dir / "heldout_validation_paths.txt"
with open(heldout_validation_paths_file, "w") as f:
    f.write("\n".join(heldout_validation_list))
print(f"Saved held-out validation file list to {heldout_validation_paths_file} ({len(heldout_validation_list)} lines)")

# Create data_oversampled.yaml
# We'll copy the class names from the original data.yaml
original_data_yaml = detector_training_data_dir / "data.yaml"
# Read the original data.yaml to get names and nc
try:
    with open(original_data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    names = data['names']
    nc = data['nc']
except Exception as e:
    print(f"Error reading original data.yaml: {e}")
    # Fallback to the known classes
    names = ['freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe']
    nc = 6

# Create the new data dictionary
data_oversampled = {
    'train': 'oversampled_image_paths.txt',  # relative to the yaml location
    'val': 'heldout_validation_paths.txt',
    'nc': nc,
    'names': names
}

# Write the new data.yaml
data_oversampled_yaml = output_dir / "data_oversampled.yaml"
with open(data_oversampled_yaml, 'w') as f:
    yaml.dump(data_oversampled, f, default_flow_style=False)
print(f"Saved data_oversampled.yaml to {data_oversampled_yaml}")

# Also, for completeness, we can save the unique source lists (not required by user)
unique_train_sources_file = output_dir / "unique_training_sources.txt"
with open(unique_train_sources_file, "w") as f:
    f.write("\n".join([str(p) for p in train_sources]))
print(f"Saved unique training sources to {unique_train_sources_file}")

unique_val_sources_file = output_dir / "unique_validation_sources.txt"
with open(unique_val_sources_file, "w") as f:
    f.write("\n".join([str(p) for p in val_sources]))
print(f"Saved unique validation sources to {unique_val_sources_file}")

print("Done.")