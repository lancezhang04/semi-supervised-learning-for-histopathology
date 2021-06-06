from tqdm import tqdm
import pandas as pd
import cv2
import os


# need to update code to handle uneven number of classes in train and test

csv_folder = 'NuCLS/csv'
image_folder = 'NuCLS/rgb'
train_split_file = 'NuCLS/train_test_splits/fold_1_train.csv'
# test_split_file = 'NuCLS/train_test_splits/fold_1_test.csv'
size = 64  # the length of the sides of the cropped out cell image, must be even
half_size = size // 2
unusable_count = 0  # number of bounding boxes that cannot be cropped into an image
usable_count = 0  # number of usable bonding boxes

all_csv = os.listdir(csv_folder)
all_csv.remove('ALL_FOV_LOCATIONS.csv')  # does not contain bbox information

train_slides = pd.read_csv(train_split_file)['slide_name'].to_list()
# test_slides = pd.read_csv(test_split_file)['slide_name'].to_list()

for csv in tqdm(all_csv, ncols=70):
    root_name = csv.split('.')[0]
    image_dir = os.path.join(image_folder, root_name + '.png')
    csv_dir = os.path.join(csv_folder, csv)

    # load image
    img = cv2.imread(image_dir)
    img_width, img_height = img.shape[1], img.shape[0]

    # load bounding box information
    df = pd.read_csv(csv_dir)
    num_bboxes = len(df)
    x_min, x_max = df['xmin'].to_list(), df['xmax'].to_list()
    y_min, y_max = df['ymin'].to_list(), df['ymax'].to_list()
    cell_group = [s[11:] if s.startswith('correction_') else s for s in df['group'].to_list()]

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

        # test if the cell has an ambiguous nuclei
        if cell_group[i] in ['unlabeled', 'correction_unlabeled', 'apoptotic_body', 'correction_apoptotic_body']:
            unusable_count += 1
            continue

        slide_name = root_name.split('_')[0]
        if slide_name in train_slides:
            split = 'train'
        else:
            split = 'test'

        cell_img = img[y_center - half_size + 1: y_center + half_size + 1, x_center - half_size + 1: x_center + half_size + 1, :]
        target_folder = f'NuCLS_{size}/{split}/{cell_group[i]}'
        if not os.path.exists(target_folder):
            # the first/second level folder needs to already exist
            os.mkdir(target_folder)

        cv2.imwrite(os.path.join(target_folder, f'{root_name}_{i}.png'), cell_img)
        usable_count += 1


print(f'Usable bounding boxes: {usable_count}\nUnusable bounding boxes: {unusable_count}\nTotal: {usable_count + unusable_count}')
print('Execution complete')
