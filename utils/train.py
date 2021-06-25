import pandas as pd
import os


def get_train_test_split(train_csv_path, test_csv_path, image_folder):
    train_csv, test_csv = pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)
    fov_names = [image_name[:-4] for image_name in os.listdir(image_folder)]
    train_fovs, test_fovs = [], []

    for _, file in train_csv.iterrows():
        slide_name = file['slide_name']
        for fov_name in fov_names:
            if slide_name in fov_name:
                train_fovs.append(fov_name)

    for _, file in test_csv.iterrows():
        slide_name = file['slide_name']
        for fov_name in fov_names:
            if slide_name in fov_name:
                test_fovs.append(fov_name)

    return train_fovs, test_fovs


def format_instance_string(instance, fov_image):
    instance_string = fov_image + ','
    instance_string += ','.join(list(str(i) for i in instance[3:7]))
    instance_string += ',' + instance['group']
    return instance_string


def generate_train_test_documents(train_csv_path, test_csv_path, image_folder, csv_folder):
    train_fovs, test_fovs = get_train_test_split(train_csv_path, test_csv_path, image_folder)
    train_document, test_document = [], []

    for fov in train_fovs:
        fov_image = os.path.join(image_folder, fov + '.png')
        fov_csv = pd.read_csv(os.path.join(csv_folder, fov + '.csv'))
        for _, cell in fov_csv.iterrows():
            train_document.append(format_instance_string(cell, fov_image))

    for fov in test_fovs:
        fov_image = os.path.join(image_folder, fov + '.png')
        fov_csv = pd.read_csv(os.path.join(csv_folder, fov + '.csv'))
        for _, cell in fov_csv.iterrows():
            test_document.append(format_instance_string(cell, fov_image))

    print('total train size:', len(train_document))
    print('total test size:', len(test_document))

    with open('../keras-frcnn/train_document.txt', 'w') as file:
        file.write('\n'.join(train_document))

    with open('../keras-frcnn/test_document.txt', 'w') as file:
        file.write('\n'.join(test_document))


if __name__ == '__main__':
    # train_fovs, test_fovs = get_train_test_split(
    #     '../NuCLS/train_test_splits/fold_1_train.csv',
    #     '../NuCLS/train_test_splits/fold_1_test.csv',
    #     '../NuCls/rgb',
    # )

    generate_train_test_documents(
        '../datasets/NuCLS/train_test_splits/fold_1_train.csv',
        '../datasets/NuCLS/train_test_splits/fold_1_test.csv',
        '../NuCls/rgb',
        '../NuCls/csv',
    )
