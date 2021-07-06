import matplotlib.pyplot as plt
from skimage import exposure
from tqdm import tqdm
import cv2
import os


SOURCE_DIR = 'NuCLS/rgb'
TARGET_DIR = 'NuCLS_histogram_matching/rgb'
REFERENCE_IMAGE_PATH = 'NuCLS/rgb/TCGA-A2-A3XS-DX1_id-5ea4096dddda5f839897afad_left-26262_top-39256_bottom-39559_right-26541.png'

SHOW_VISUALIZATION = True

os.makedirs(TARGET_DIR, exist_ok=True)
ref_image = cv2.imread(REFERENCE_IMAGE_PATH)


for src_image_name in tqdm(os.listdir(SOURCE_DIR), ncols=100):
    src_image = cv2.imread(os.path.join(SOURCE_DIR, src_image_name))
    matched_image = exposure.match_histograms(src_image, ref_image)

    if SHOW_VISUALIZATION:
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))

        for i, im in enumerate([ref_image, src_image, matched_image]):
            axs[i].axis('off')
            axs[i].imshow(im.astype(int))

        plt.show()
    else:
        cv2.imwrite(os.path.join(TARGET_DIR, src_image_name), matched_image)
