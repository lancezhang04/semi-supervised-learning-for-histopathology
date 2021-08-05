DATASETS_CONFIG = {}

# cell classification
DATASETS_CONFIG['cell_7_0.5'] = {
    'type': 'cell',

    'split_file_path': 'datasets/NuCLS/train_test_splits/fold_1_test.csv',
    'dataset_dir': 'datasets/NuCLS/NuCLS_64',

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
}

# tissue classification
DATASETS_CONFIG['tissue_5_0.5'] = {
    'type': 'tissue',

    'split_file_path': 'datasets/tissue_classification/fold_test.csv',
    'dataset_dir': 'datasets/tissue_classification/dataset',

    'groups': {
        'TUMOR': 'tumor',
        'STROMA': 'stroma',
        'TILS': 'tils',
        'JUNK': 'junk',
        'WHITE': 'white'
    },
    'major_groups': ['tumor', 'stroma']
}

DATASETS_CONFIG['tissue_8_0.3'] = {
    'type': 'tissue',

    'split_file_path': 'datasets/tissue_classification/fold_test.csv',
    'dataset_dir': 'datasets/tissue_classification/dataset_main_0.3',

    'groups': {
        'Tumor': 'tumor',
        'Fat': 'fat',
        'NecroticDebris': 'necrotic_debris',
        'TILsDense': 'tils',
        'StromaNOS': 'stroma',

        'Blood': 'blood',
        'BloodVessel': 'blood_vessel',
        'PlasmaCellInfiltrate': 'plasma_cell_infiltrate'
    },
    'major_groups': ['tumor', 'stroma']
}
