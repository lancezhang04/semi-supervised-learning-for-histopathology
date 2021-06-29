from silence_tensorflow import silence_tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pickle

silence_tensorflow()


# ==================================================================================================================== #
# Configuration
# ==================================================================================================================== #
GROUP_BY = ['cell_type', 'train_test', 'hospital', 'patient']
GROUP_BY = GROUP_BY[0]

DATASET_DIR = '../datasets/NuCLS_64_7_grouped/train' if GROUP_BY != 'train_test' else 'datasets/NuCLS_64_7_grouped'
NUM_CLASSES_SHOWN = None

EMBEDDINGS_PATH = 'cache/embeddings_train_barlow_fine_tune_0.01.pickle'
LABELS_PATH = 'cache/labels_cell_type.pickle'
MODEL_PATH = '../trained_models/resnet_encoders/1024/resnet_enc_barlow_fine_tune_0.01.h5'
# MODEL_PATH = 'trained_models/resnet_encoders/encoder_1024_47.02.h5'

EMBEDDINGS_SAVE_PATH = 'embeddings.pickle'
LABELS_SAVE_PATH = 'labels.pickle'

COLORS = ['red', 'blue'] + \
         [plt.cm.Set1(i) for i in range(8)] + \
         [plt.cm.Set2(i) for i in range(9)] + \
         [plt.cm.tab20(i) for i in range(20)] + \
         [plt.cm.tab20b(i) for i in range(20)] + \
         [plt.cm.tab20c(i) for i in range(20)] + \
         [plt.cm.Pastel1(i) for i in range(9)] + \
         [plt.cm.Accent(i) for i in range(8)] + \
         [plt.cm.Dark2(i) for i in range(8)]
NUM_COLORS = len(COLORS)


# ==================================================================================================================== #
# Load data
# ==================================================================================================================== #
datagen = ImageDataGenerator()
datagen = datagen.flow_from_directory(
    DATASET_DIR, shuffle=False, target_size=(64, 64), batch_size=32
)


# ==================================================================================================================== #
# Create/load embeddings
# ==================================================================================================================== #
if EMBEDDINGS_PATH is None:
    # Generate features
    print('Generating features')
    from utils.models import resnet20

    resnet_enc = resnet20.get_network(
        hidden_dim=1024,
        use_pred=False,
        return_before_head=False,
        input_shape=(64, 64, 3)
    )
    resnet_enc.load_weights(MODEL_PATH)

    preds = resnet_enc.predict(datagen, verbose=1)

    # Reduce dimensionality using TSNE
    print('Reducing dimensionality, this may take a while')
    preds_embedded = TSNE(n_components=2).fit_transform(preds)

    # Save embeddings
    print('Saving embeddings')
    with open(EMBEDDINGS_SAVE_PATH, 'wb') as file:
        pickle.dump(preds_embedded, file)
else:
    with open(EMBEDDINGS_PATH, 'rb') as file:
        preds_embedded = pickle.load(file)


# ==================================================================================================================== #
# Create/load labels
# ==================================================================================================================== #
if LABELS_PATH is None:
    # Retrieve labels for color coding
    print('Retrieving labels')
    if GROUP_BY == 'cell_type' or GROUP_BY == 'train_test':
        all_labels = datagen.labels
    elif GROUP_BY == 'hospital':
        # Group by hospital
        all_labels = [f.split('-')[1] for f in datagen.filenames]
    elif GROUP_BY == 'patient':
        # Group by patients
        all_labels = [f.split('-')[2] for f in datagen.filenames]
    else:
        raise ValueError

    # Save labels
    print('Saving labels')
    with open(LABELS_SAVE_PATH, 'wb') as file:
        pickle.dump(all_labels, file)
else:
    with open(LABELS_PATH, 'rb') as file:
        all_labels = pickle.load(file)


# ==================================================================================================================== #
# Plot results
# ==================================================================================================================== #
all_labels = np.array(all_labels)
classes = np.unique(all_labels)
classes = classes[:NUM_CLASSES_SHOWN if NUM_CLASSES_SHOWN else len(classes)]

num_classes = len(classes)
print('num_classes', num_classes)
assert num_classes <= NUM_COLORS

if GROUP_BY == 'cell_type' or GROUP_BY == 'train_test':
    class_indices = {v: k for k, v in datagen.class_indices.items()}
else:
    class_indices = dict(zip([i for i in range(num_classes)], classes))

plt.figure(figsize=(15, 10))

for g, group in enumerate(classes):
    ix = np.where(all_labels == group)
    plt.scatter(preds_embedded[ix, 0], preds_embedded[ix, 1], alpha=0.15, color=COLORS[g], label=class_indices[g])

if GROUP_BY != 'patient':
    # Grouping by patient results in too many classes
    plt.legend()
plt.show()
