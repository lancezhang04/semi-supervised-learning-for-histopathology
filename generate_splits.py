from collections import defaultdict
import numpy as np
import os


TEST_SPLIT = 0.15
ROOT_DIR = 'tissue_classification/rgbs_colorNormalized'
RANDOM_SEED = 42
# ROOT_DIR = '../datasets/NuCLS/rgb'


def count_hospitals():
    hospital_counts = defaultdict(lambda: 0)
    total = 0

    for slide_name in os.listdir(ROOT_DIR):
        hospital_counts[slide_name.split('-')[1]] += 1
        total += 1

    return dict(hospital_counts), total


def generate_splits():
    hospital_counts, total = count_hospitals()

    num_test = int(total * TEST_SPLIT)
    train_hospitals, test_hospitals = [], []

    count = 0
    hospitals = list(hospital_counts.keys())

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(hospitals)

    for hospital in hospitals:
        if count < num_test:
            count += hospital_counts[hospital]
            test_hospitals.append(hospital)
        else:
            train_hospitals.append(hospital)

    return train_hospitals, test_hospitals


if __name__ == '__main__':
    # NuCLS uses data from 124/125 patients whereras the tissue classification dataset contains 151 patients
    train_h, test_h = generate_splits()
    print(train_h, test_h)

    # Generate a split file
    import pandas as pd

    df = {'slide_name': [], 'hospital': []}

    for fov in os.listdir(ROOT_DIR):
        slide_name = fov.split('_')[0]
        hospital = slide_name.split('-')[1]

        if (slide_name not in df['slide_name']) and (hospital in test_h):
            df['slide_name'].append(slide_name)
            df['hospital'].append(hospital)

    df = pd.DataFrame(df)
    df['split'] = 'test'

    df.to_csv('fold_test.csv')
