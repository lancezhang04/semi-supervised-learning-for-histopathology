from utils.detect_whitespace import detect_whitespace
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import cv2
import os


def determine_patch_class(patch_mask, classes_map):
    unique, counts = np.unique(patch_mask, return_counts=True)
    ratios = counts / (PATCH_SIZE ** 2)
    max_ratio = max(ratios)

    if max_ratio >= THRESHOLD:
        tissue_type = unique[np.argmax(ratios)]
        tissue_type = classes_map[tissue_type]

        if not INCLUDE_EXCLUDE and tissue_type.lower() == 'exclude':
            tissue_type = None
    else:
        tissue_type = None

    return tissue_type, round(max_ratio, 3)


def generate_patches(img, mask, img_name, classes_map):
    horizontal_steps = int((img.shape[1] - PATCH_SIZE) / STEP_SIZE)
    vertical_steps = int((img.shape[0] - PATCH_SIZE) / STEP_SIZE)

    x_min, y_min = 0, 0
    patches_generated = 0
    tissue_type_counts = defaultdict(lambda: 0)

    for y_step in range(vertical_steps):
        for x_step in range(horizontal_steps):
            # Cut out the patch from the image
            x_max, y_max = int(x_min + PATCH_SIZE), int(y_min + PATCH_SIZE)
            patch = img[y_min: y_max, x_min: x_max, :]
            patch_mask = mask[y_min: y_max, x_min: x_max, 0]

            if GENERATE_ENCODER_DATASET:
                # Save patch for encoder, all image files are saved in the 'all' folder
                cv2.imwrite(os.path.join(ENCODER_TARGET_DIR, 'all', f'{img_name}_{x_step}_{y_step}.png'), patch)

            # Determine patch class
            tissue_type, max_ratio = determine_patch_class(patch_mask, classes_map)
            if tissue_type is not None:
                patch_save_path = os.path.join(
                    TARGET_DIR, tissue_type, f'{img_name}_{patches_generated}_{max_ratio}.png')
                os.makedirs(os.path.dirname(patch_save_path), exist_ok=True)
                cv2.imwrite(patch_save_path, patch)

                patches_generated += 1
                tissue_type_counts[tissue_type] += 1

            x_min += STEP_SIZE

        x_min = 0
        y_min += STEP_SIZE

    return patches_generated, tissue_type_counts


def main():
    # Create dataset directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    if GENERATE_ENCODER_DATASET:
        os.makedirs(os.path.join(ENCODER_TARGET_DIR, 'all'), exist_ok=True)

    df = pd.read_csv('datasets/tissue_classification/region_GTcodes.csv', delimiter=',')
    classes_map = dict(zip(df[CLASSES_MODE + '_codes'], df[CLASSES_MODE + '_classes']))
    gt_codes_map = dict(zip(df['GT_code'], df[CLASSES_MODE + '_codes']))

    # For later usage on mask
    k, v = np.array(list(gt_codes_map.keys())), np.array(list(gt_codes_map.values()))
    sort_idx = k.argsort()
    k_sorted, v_sorted = k[sort_idx], v[sort_idx]

    total_patches_generated = 0
    total_tissue_type_counts = defaultdict(lambda: 0)

    for image_name, mask_name in tqdm(zip(os.listdir(RGB_DIR), os.listdir(MASKS_DIR)),
                                      total=len(os.listdir(RGB_DIR)), ncols=100):
        image = cv2.imread(os.path.join(RGB_DIR, image_name))
        mask = cv2.imread(os.path.join(MASKS_DIR, mask_name))

        # Detect whitespace and apply (grouped) label
        mask = v_sorted[np.searchsorted(k_sorted, mask)]  # this needs to happen FIRST!
        mask = detect_whitespace(image, mask, WHITESPACE_CODE)

        patches_generated, tissue_type_counts = generate_patches(
            image, mask,
            image_name.split('.')[0],
            classes_map
        )

        total_patches_generated += patches_generated
        for k, v in tissue_type_counts.items():
            total_tissue_type_counts[k] += v

    total_tissue_type_counts = dict(total_tissue_type_counts)
    print('total patches generated:', total_patches_generated)

    with open(os.path.join(TARGET_DIR, 'meta.json'), 'w') as file:
        json.dump(total_tissue_type_counts, file)
    print(total_tissue_type_counts)


if __name__ == '__main__':
    MASKS_DIR = 'datasets/tissue_classification/masks'
    RGB_DIR = 'datasets/tissue_classification/rgbs_colorNormalized'

    PATCH_SIZE = 224
    STEP_SIZE = int(0.5 * PATCH_SIZE)
    THRESHOLD = 0.3
    INCLUDE_EXCLUDE = False

    TARGET_DIR = 'datasets/tissue_classification/dataset_main_classes_0.3'
    GENERATE_ENCODER_DATASET = False
    ENCODER_TARGET_DIR = 'datasets/tissue_classification/dataset_encoder'

    CLASSES_MODE = ['main', 'super'][0]  # use the `main_classes` or `super_classes` column
    WHITESPACE_CODE = 8 if CLASSES_MODE == 'super' else 11

    print('Saving dataset at:', TARGET_DIR)
    main()
