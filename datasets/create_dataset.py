from tqdm import tqdm
import pandas as pd
import cv2
import os


SOURCE_DIR = 'NuCLS_histogram_matching'
SIZE = 64  # the length of the sides of the cropped out cell image, must be even
TARGET_DIR = f'NuCLS_histogram_matching/NuCLS_histogram_matching_{SIZE}'
print(SOURCE_DIR, '>', TARGET_DIR)

os.makedirs(TARGET_DIR, exist_ok=True)

csv_dir = os.path.join('NuCLS', 'csv')
image_dir = os.path.join(SOURCE_DIR, 'rgb')
train_split_file = os.path.join('NuCLS', 'train_test_splits/fold_1_train.csv')

classes_used = [
    'tumor',
    'fibroblast',
    'lymphocyte',
    'plasma_cell',
    'macrophage',
    'apoptotic_body',
    'vascular_endothelium'
]

half_size = SIZE // 2
unusable_count = 0  # number of bounding boxes that cannot be cropped into an image
usable_count = 0  # number of usable bonding boxes

all_csv = os.listdir(csv_dir)
all_csv.remove('ALL_FOV_LOCATIONS.csv')  # does not contain bbox information


for csv in tqdm(all_csv, ncols=100):
    root_name = csv.split('.')[0]
    image_path = os.path.join(image_dir, root_name + '.png')
    csv_path = os.path.join(csv_dir, csv)

    # load image
    img = cv2.imread(image_path)
    img_width, img_height = img.shape[1], img.shape[0]

    # load bounding box information
    df = pd.read_csv(csv_path)
    num_bboxes = len(df)
    x_min, x_max = df['xmin'].to_list(), df['xmax'].to_list()
    y_min, y_max = df['ymin'].to_list(), df['ymax'].to_list()
    cell_classes = [s[11:] if s.startswith('correction_') else s for s in df['group'].to_list()]

    for i in range(num_bboxes):
        x_center = int((x_max[i] + x_min[i]) / 2)
        y_center = int((y_max[i] + y_min[i]) / 2)

        # test if the resulting image goes out of bound ==> makes bbox unusable
        # when cropping, 31 pixels are taken from the left side of the center and 32 pixels from the right side
        if (x_center < half_size - 1) or (y_center < half_size - 1):
            unusable_count += 1
            continue
        if (x_center + half_size + 1 > img_width + 1) or (y_center + half_size + 1 > img_height + 1):
            unusable_count += 1
            continue

        # test if the cell belongs to the classes used
        cell_class = cell_classes[i]
        if cell_class not in classes_used:
            unusable_count += 1
            continue

        cell_img = img[y_center - half_size + 1: y_center + half_size + 1,
                       x_center - half_size + 1: x_center + half_size + 1, :]

        folder_name = cell_class
        target_folder = os.path.join(TARGET_DIR, folder_name)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        cv2.imwrite(os.path.join(target_folder, f'{root_name}_{i}.png'), cell_img)
        usable_count += 1


print(f'Usable bounding boxes: {usable_count}\nUnusable bounding boxes: {unusable_count}\nTotal: {usable_count + unusable_count}')
print('Execution complete')
