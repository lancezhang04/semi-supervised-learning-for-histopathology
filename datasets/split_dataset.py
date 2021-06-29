from shutil import copyfile
from pathlib import Path
import numpy as np
import os


SPLIT = 0.5  # How much of the original dataset to use as training data
VALIDATION_SPLIT = 0.15  # How much of the original dataset to use as validation data
assert (SPLIT + VALIDATION_SPLIT) <= 1
ROOT_DIR = 'NuCLS_64_7_grouped/train'  # The directory of training set of the original dataset

TARGET_DIR = f'datasets/{ROOT_DIR.split("/")[1]}_{SPLIT}'
TRAIN_DIR = os.path.join(TARGET_DIR, 'train')
VAL_DIR = os.path.join(TARGET_DIR, 'val')
RANDOM_SEED = 42

Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
Path(VAL_DIR).mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)
log = ''


for cell_type in os.listdir(ROOT_DIR):
    cell_dir = os.path.join(ROOT_DIR, cell_type)
    cell_train_dir = os.path.join(TRAIN_DIR, cell_type)
    cell_val_dir = os.path.join(VAL_DIR, cell_type)

    Path(cell_train_dir).mkdir(exist_ok=False)
    Path(cell_val_dir).mkdir(exist_ok=False)

    cell_images = os.listdir(cell_dir)
    num_cell_images = len(cell_images)

    num_total_selected = round(num_cell_images * (SPLIT + VALIDATION_SPLIT))
    num_train_selected = round(num_cell_images * SPLIT)

    selected = np.random.choice(cell_images, num_total_selected, replace=False)

    for cell_image in selected[:num_train_selected]:
        copyfile(os.path.join(cell_dir, cell_image), os.path.join(cell_train_dir, cell_image))
    for cell_image in selected[num_train_selected:]:
        copyfile(os.path.join(cell_dir, cell_image), os.path.join(cell_val_dir, cell_image))

    log_message = f'{cell_type}, train: {num_train_selected}, val: {num_total_selected - num_train_selected}'
    log += log_message + '\n'
    print(log_message)


with open(os.path.join(TARGET_DIR, 'log.txt'), 'w') as file:
    file.write(log)
