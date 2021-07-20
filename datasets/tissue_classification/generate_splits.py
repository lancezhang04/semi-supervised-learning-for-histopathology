from collections import defaultdict
import pandas as pd
import numpy as np
import os


def count_hospitals():
    hospital_counts = defaultdict(lambda: 0)
    total = 0

    for slide_name in os.listdir(ROOT_DIR):
        hospital_counts[slide_name.split('-')[1]] += 1
        total += 1

    return dict(hospital_counts), total


def generate_splits():
    assert NUM_FOLDS * TEST_SPLIT <= 1

    hospital_counts, total = count_hospitals()

    num_test = int(total * TEST_SPLIT)
    folds = [[] for i in range(NUM_FOLDS)]
    fold_counts = []

    count = 0
    fold_idx = 0
    hospitals = list(hospital_counts.keys())

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(hospitals)

    for hospital in hospitals:
        count += hospital_counts[hospital]
        folds[fold_idx].append(hospital)

        if count >= num_test:
            fold_counts.append(count)
            fold_idx += 1
            count = 0

        if fold_idx == NUM_FOLDS:
            break

    return folds, fold_counts


def save_splits(folds, counts):
    meta_file = open(os.path.join(SAVE_DIR, 'meta.txt'), 'w')
    for i, fold in enumerate(folds):
        df = {'slide_name': [], 'hospital': []}

        for fov in os.listdir(ROOT_DIR):
            slide_name = fov.split('_')[0]
            hospital = slide_name.split('-')[1]

            if (slide_name not in df['slide_name']) and (hospital in folds[i]):
                df['slide_name'].append(slide_name)
                df['hospital'].append(hospital)

        df = pd.DataFrame(df)
        df['split'] = 'test'

        df.to_csv(os.path.join(SAVE_DIR, f'test_fold_{i}.csv'))

        meta_file.write(str(counts[i]) + ' - ' + ', '.join(fold) + '\n')

    meta_file.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    folds, counts = generate_splits()
    save_splits(folds, counts)


if __name__ == '__main__':
    TEST_SPLIT = 0.15
    NUM_FOLDS = 6
    ROOT_DIR = '../tissue_classification/rgbs_colorNormalized'
    SAVE_DIR = '../tissue_classification/splits'
    RANDOM_SEED = 42

    main()
