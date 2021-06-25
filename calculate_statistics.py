import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os


count_group_and_type = True
show_box_sizes = False

if count_group_and_type:
    csv_folder = 'NuCLS/csv'
    all_csv = os.listdir(csv_folder)
    all_csv.remove('ALL_FOV_LOCATIONS.csv')

    print('Total FOVs', len(all_csv))

    type_count = dict()
    group_count = dict()

    for csv_file in tqdm(all_csv):
        fov = pandas.read_csv(os.path.join(csv_folder, csv_file))
        for cell in fov.iterrows():
            cell = cell[1]  # `cell[0]` is the index
            if cell['group'] in group_count:
                group_count[cell['group']] += 1
            else:
                group_count[cell['group']] = 1

            if cell['type'] in type_count:
                type_count[cell['type']] += 1
            else:
                type_count[cell['type']] = 1

    print('type_count:', type_count)
    print('group_count:', group_count)

if show_box_sizes:
    csv_dir = 'datasets/NuCLS/csv'
    all_csv = os.listdir(csv_dir)
    all_csv.remove('ALL_FOV_LOCATIONS.csv')

    width_values, height_values = [], []

    for csv in tqdm(all_csv, ncols=70):
        df = pd.read_csv(os.path.join(csv_dir, csv))
        x_min, x_max = df['xmin'].to_list(), df['xmax'].to_list()
        width_values.extend([x_max[i] - x_min[i] for i in range(len(x_max))])

        y_min, y_max = df['ymin'].to_list(), df['ymax'].to_list()
        height_values.extend([y_max[i] - y_min[i] for i in range(len(y_max))])

    plt.hist(width_values, bins=30, color='orange')
    plt.show()

    plt.hist(height_values, bins=30, color='orange')
    plt.show()

