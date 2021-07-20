from utils.detect_whitespace import detect_whitespace
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os


def generate_patches(img, mask, img_name, classes_map, verbose=0):
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
            assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)

            if GENERATE_ENCODER_DATASET:
                # Save patch for encoder, all image files are saved in the 'all' folder
                cv2.imwrite(os.path.join(ENCODER_TARGET_DIR, 'all', f'{img_name}_{x_step}_{y_step}.png'), patch)

            # Determine patch class
            patch_mask = mask[y_min: y_max, x_min: x_max, 0]
            unique, counts = np.unique(patch_mask, return_counts=True)
            ratios = counts / (PATCH_SIZE ** 2)

            if max(ratios) >= THRESHOLD:
                tissue_type = unique[np.argmax(ratios)]
                tissue_type = classes_map[tissue_type]
                if not INCLUDE_EXCLUDE and tissue_type.lower() == 'exclude':
                    continue
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


def main():
    df = pd.read_csv('tissue_classification/region_GTcodes.csv', delimiter=',')

    # Maps class code ==> class name
    classes_map = dict(zip(df[CLASSES_MODE + '_codes'], df[CLASSES_MODE + '_classes']))
    # Maps ground truth codes (every class) ==> class codes (every main/super class)
    gt_codes_map = dict(zip(df['GT_code'], df[CLASSES_MODE + '_codes']))

    # For later usage on mask
    k, v = np.array(list(gt_codes_map.keys())), np.array(list(gt_codes_map.values()))
    sort_idx = k.argsort()
    k_sorted, v_sorted = k[sort_idx], v[sort_idx]

    # Create dataset directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    if GENERATE_ENCODER_DATASET:
        os.makedirs(os.path.join(ENCODER_TARGET_DIR, 'all'), exist_ok=True)

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
            classes_map,
            verbose=0
        )

        total_patches_generated += patches_generated
        for k, v in tissue_type_counts.items():
            total_tissue_type_counts[k] += v

    total_tissue_type_counts = dict(total_tissue_type_counts)
    print('total patches generated:', total_patches_generated)
    print(total_tissue_type_counts)


if __name__ == '__main__':
    MASKS_DIR = 'tissue_classification/masks'
    RGB_DIR = 'tissue_classification/rgbs_colorNormalized'

    TARGET_DIR = 'tissue_classification/dataset_main_classes'
    PATCH_SIZE = 224
    STEP_SIZE = int(0.5 * PATCH_SIZE)
    THRESHOLD = 0.5
    INCLUDE_EXCLUDE = True

    GENERATE_ENCODER_DATASET = False
    ENCODER_TARGET_DIR = 'tissue_classification/dataset_encoder'

    # Change code in detect_white_space() when using main classes
    CLASSES_MODE = ['main', 'super'][0]  # use `main_classes` or `super_classes`
    WHITESPACE_CODE = 8 if CLASSES_MODE == 'super' else 11

    main()
