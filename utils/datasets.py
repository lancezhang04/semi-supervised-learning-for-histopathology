from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os


def get_dataset_df(config, random_seed):
    np.random.seed(random_seed)

    # Generate DataFrame from dataset
    df = {
        'filename': [],
        'class': [],
        'slide_name': [],
        'hospital': [],
        'patient': []
    }

    for root, dirs, files in os.walk(config['dataset_dir']):
        for name in files:
            path = os.path.join(root, name)
            dir_, file = os.path.split(path)

            df['filename'].append(path)
            df['class'].append(os.path.split(dir_)[1])

            slide_name = file.split('_')[0]
            df['slide_name'].append(slide_name)
            df['hospital'].append(slide_name.split('-')[1])
            df['patient'].append(slide_name.split('-')[2])

    df = pd.DataFrame(df)

    # Group the cells
    for k, v in config['groups'].items():
        df.loc[df['class'] == k, 'class'] = v

    # Generate minor/major classes
    df['evaluation'] = 'minor'
    df.loc[df['class'].isin(config['major_groups']), 'evaluation'] = 'major'

    # Generate splits in the dataset
    df['split'] = 'left_out'
    split = config['split']

    if type(split) == str:
        test_slides = list(pd.read_csv(split)['slide_name'])
        df.loc[df['slide_name'].isin(test_slides), 'split'] = 'test'
    else:
        df.loc[df['hospital'].isin(split), 'split'] = 'test'

    class_counts = dict(df['class'].value_counts())
    train_split = config['train_split']
    validation_split = config['validation_split']

    for class_ in class_counts.keys():
        idxs = np.where((df['class'] == class_) & (df['split'] == 'left_out'))[0]
        num_total = int(round(len(idxs)) * (train_split + validation_split))
        num_val = int(round(len(idxs) * validation_split))

        idxs = np.random.choice(idxs, num_total, replace=False)

        # Ensures that the validation set is the same every time
        df.loc[idxs[:num_val], 'split'] = 'val'
        df.loc[idxs[num_val:], 'split'] = 'train'

    return df


def get_generators(splits, image_shape, batch_size,
                   random_seed, config=None, df=None, separate_evaluation_groups=False,
                   y_col='class', shuffle=True):
    if df is None:
        df = get_dataset_df(config, random_seed)

    generators = []
    for split in splits:
        if not separate_evaluation_groups:
            datagen = ImageDataGenerator().flow_from_dataframe(
                df[df['split'] == split], seed=random_seed,
                target_size=image_shape[:2], batch_size=batch_size,
                y_col=y_col, shuffle=shuffle
            )
            generators.append(datagen)
        else:
            generators.append([])
            for evaluation_group in ['minor', 'major']:
                print(evaluation_group + ': ', end='')
                datagen = ImageDataGenerator().flow_from_dataframe(
                    df[(df['split'] == split) & (df['evaluation'] == evaluation_group)], seed=random_seed,
                    target_size=image_shape[:2], batch_size=batch_size, y_col=y_col,
                    classes=list(np.unique(df['class'])), shuffle=shuffle
                )
                generators[-1].append(datagen)

    return generators
