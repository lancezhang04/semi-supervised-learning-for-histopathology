from collections import defaultdict
from tissue_classification.generate_splits import generate_splits
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import os


MASKS_DIR = 'masks'
RGB_DIR = 'rgbs_colorNormalized'
TARGET_DIR = 'tissue_classification'

PATCH_SIZE = 224
STEP_SIZE = int(0.5 * PATCH_SIZE)

df = pd.read_csv('meta/gtruth_codes.tsv', delimiter='\t')
GT_CODES = dict(zip(df['GT_code'], df['label']))


os.makedirs(TARGET_DIR, exist_ok=True)


def generate_patches(img, mask, img_name, verbose=0):
    horizontal_steps = int((img.shape[1] - PATCH_SIZE) / STEP_SIZE)
    vertical_steps = int((img.shape[0] - PATCH_SIZE) / STEP_SIZE)

    if verbose:
        print('hotizontal_step:', horizontal_steps, '\nvertical steps:', vertical_steps)
        print('total patches possible:', horizontal_steps * vertical_steps)

    x_min, y_min = 0, 0
    patches_generated = 0
    tissue_type_counts = defaultdict(lambda: 0)

    for y_step in range(vertical_steps):
        for x_step in range(horizontal_steps):
            x_max, y_max = int(x_min + PATCH_SIZE), int(y_min + PATCH_SIZE)

            patch = img[y_min: y_max, x_min: x_max, :]
            patch_mask = mask[y_min: y_max, x_min: x_max, 0]
            assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)

            unique, counts = np.unique(patch_mask, return_counts=True)
            ratios = counts / (PATCH_SIZE ** 2)

            if max(ratios) >= 0.5:
                tissue_type = unique[np.argmax(ratios)]
                tissue_type = GT_CODES[tissue_type]
            else:
                continue

            patch_save_dir = os.path.join(TARGET_DIR, tissue_type)
            os.makedirs(patch_save_dir, exist_ok=True)

            cv2.imwrite(os.path.join(patch_save_dir, f'{img_name}_{patches_generated}_{round(max(ratios), 3)}.png'),
                        patch)

            patches_generated += 1
            tissue_type_counts[tissue_type] += 1
            x_min += STEP_SIZE

        x_min = 0
        y_min += STEP_SIZE

    if verbose:
        print('patches_generated:', patches_generated)
        print(dict(tissue_type_counts))

    return patches_generated, tissue_type_counts


if __name__ == '__main__':
    total_patches_generated = 0
    total_tissue_type_counts = defaultdict(lambda: 0)

    for image_name, mask_name in tqdm(zip(os.listdir(RGB_DIR), os.listdir(MASKS_DIR)),
                                      total=len(os.listdir(RGB_DIR)), ncols=100):
        image = cv2.imread(os.path.join(RGB_DIR, image_name))
        mask = cv2.imread(os.path.join(MASKS_DIR, mask_name))

        patches_generated, tissue_type_counts = generate_patches(
            image, mask,
            image_name.split('.')[0],
            verbose=0
        )

        total_patches_generated += patches_generated
        for k, v in tissue_type_counts.items():
            total_tissue_type_counts[k] += v

    total_tissue_type_counts = dict(total_tissue_type_counts)
    print('total patches generated:', total_patches_generated)
    print(total_tissue_type_counts)
