from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from utils.datasets import get_dataset_df, get_generators
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pickle
import os


# ==================================================================================================================== #
# Configuration
# ==================================================================================================================== #
GROUP_BY = ['cell_type', 'train_test', 'hospital', 'patient']
GROUP_BY = GROUP_BY[0]
USE_SPLIT = 'all'

IMAGE_SHAPE = (64, 64, 3)
DATASET_CONFIG = {
    'split': '../datasets/NuCLS/train_test_splits/fold_1_test.csv',
    'train_split': 0.1,
    'validation_split': 0.15,
    'dataset_dir': '../datasets/NuCLS_64',
    'cell_groups': {
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

EMBEDDINGS_PATH = None  # 'embeddings.pickle'
EMBEDDINGS_SAVE_PATH = 'embeddings.pickle'

# MODEL_PATH = '../trained_models/resnet_encoders/1024/resnet_enc_barlow_fine_tune_0.85.h5'
# MODEL_PATH = 'trained_models/resnet_encoders/encoder_1024_47.02.h5'
MODEL_PATH = '../trained_models/resnet_encoders/1024/resnet_enc_supervised_0.85.h5'

COLORS = [plt.cm.Set1(i) for i in range(8)] + \
         [plt.cm.Set2(i) for i in range(9)] + \
         [plt.cm.tab20(i) for i in range(20)] + \
         [plt.cm.tab20b(i) for i in range(20)] + \
         [plt.cm.tab20c(i) for i in range(20)] + \
         [plt.cm.Pastel1(i) for i in range(9)] + \
         [plt.cm.Accent(i) for i in range(8)] + \
         [plt.cm.Dark2(i) for i in range(8)]


# ==================================================================================================================== #
# Load data
# ==================================================================================================================== #
dataset_df = get_dataset_df(DATASET_CONFIG, 42)
if USE_SPLIT == 'all':
    dataset_df = dataset_df[dataset_df['split'] != 'left_out']
    dataset_df['split_label'] = dataset_df['split']
    dataset_df['split'] = 'all'
# else:
#     dataset_df = dataset_df[dataset_df['split'] == 'left_out']

y_col = {
    'cell_type': 'class',
    'train_test': 'split_label',
    'hospital': 'hospital',
    'patient': 'patient'
}

datagen = get_generators(
    [USE_SPLIT], IMAGE_SHAPE, 32,
    dataset_config=None, random_seed=42, df=dataset_df,
    y_col=y_col[GROUP_BY], shuffle=False
)[0]


# ==================================================================================================================== #
# Create/load embeddings
# ==================================================================================================================== #
if EMBEDDINGS_PATH is None:
    # Generate features
    print('Generating features...')
    from utils.models import resnet_cifar

    resnet_enc = resnet_cifar.get_network(
        hidden_dim=1024,
        use_pred=False,
        return_before_head=False,
        input_shape=IMAGE_SHAPE
    )
    resnet_enc.load_weights(MODEL_PATH)

    preds = resnet_enc.predict(datagen, verbose=1)

    # Reduce dimensionality using TSNE
    print('Reducing dimensionality, this may take a while...')
    preds_embedded = TSNE(n_components=2).fit_transform(preds)

    # Save embeddings
    print('Saving embeddings')
    with open(EMBEDDINGS_SAVE_PATH, 'wb') as file:
        pickle.dump(preds_embedded, file)
else:
    with open(EMBEDDINGS_PATH, 'rb') as file:
        preds_embedded = pickle.load(file)


# ==================================================================================================================== #
# Plot results
# ==================================================================================================================== #
all_labels = np.array(datagen.labels)

classes = np.unique(all_labels)
if GROUP_BY == 'patient':
    classes = classes[:len(COLORS)]

num_classes = len(classes)
class_indices = {v: k for k, v in datagen.class_indices.items()}


plt.figure(figsize=(15, 10))
plt.title(os.path.basename(MODEL_PATH))

for g, group in enumerate(classes):
    ix = np.where(all_labels == group)
    plt.scatter(preds_embedded[ix, 0], preds_embedded[ix, 1],
                alpha=0.25, color=COLORS[g], s=60,
                label=class_indices[g])

if GROUP_BY != 'patient':
    # Grouping by patient results in too many classes
    plt.legend()
plt.show()
