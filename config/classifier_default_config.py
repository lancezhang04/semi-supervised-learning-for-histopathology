MODEL_TYPE = ['supervised', 'barlow', 'barlow_fine_tuned'][2]
PROJECTOR_DIMENSIONALITY = 512
IMAGE_SHAPE = [32, 32, 3]

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
PATIENCE = 100
EPOCHS = 30
PREFETCH = 8

RANDOM_SEED = 42

PRETRAINED_DIR = None
ROOT_SAVE_DIR = 'trained_models/classifiers'

# ==================================================================================================================== #
# Configuration - dataset
# ==================================================================================================================== #

# cell classification
'''DATASET_CONFIG = {
    'type': 'cell',

    'split': 'datasets/NuCLS/train_test_splits/fold_1_test.csv',
    'train_split': 0.5,
    'validation_split': 0.15,
    'dataset_dir': 'datasets/NuCLS_histogram_matching/NuCLS_histogram_matching_64',
    'groups': {
        'tumor': 'tumor',
        'fibroblast': 'stromal',
        'vascular_endothelium': 'vascular_endothelium',
        'macrophage': 'stromal',
        'lymphocyte': 'stils',
        'plasma_cell': 'stils',
        'apoptotic_body': 'apoptotic_body'
    },
    'major_groups': ['tumor', 'stils']
}'''

# tissue classification
DATASET_CONFIG = {
    'type': 'tissue',

    'split_file_path': 'tissue_classification/fold_test.csv',
    'train_split': 0.5,
    'validation_split': 0.15,
    'dataset_dir': 'tissue_classification/dataset',
    'groups': {
        'TUMOR': 'tumor',
        'STROMA': 'stroma',
        'TILS': 'tils',
        'JUNK': 'junk'
    },
    'major_groups': ['tumor', 'stroma']
}
