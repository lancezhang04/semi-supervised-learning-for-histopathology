import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import cv2


def draw_image(path):
    img = cv2.imread(path)
    img = np.array(img)
    plt.imshow(img)


def draw_box(x_min, y_min, x_max, y_max):
    rect = patches.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min), alpha=1, fill=False, edgecolor='r')
    plt.gca().add_patch(rect)


def display_image(path):
    draw_image(path)
    plt.show()


def display_image_with_box(path, x_min, y_min, x_max, y_max):
    draw_image(path)
    draw_box(x_min, y_min, x_max, y_max)
    plt.show()


def visualize_image(image_path, csv_path):
    draw_image(image_path)
    df = pd.read_csv(csv_path)
    for _, cell in df.iterrows():
        draw_box(cell['xmin'], cell['ymin'], cell['xmax'], cell['ymax'])
        plt.gca().annotate(cell['group'], xy=[cell['xmin'] , (cell['ymax'] + cell['ymin']) / 2], c='w')
    plt.show()


if __name__ == '__main__':
    image_path = '../NuCLS/rgb/TCGA-A2-A0SX-DX1_id-5ea40b05ddda5f839899743b_left-54812_top-58969_bottom-59272_right-55081.png'
    file_path = '../NuCLS/csv/TCGA-A2-A0SX-DX1_id-5ea40b05ddda5f839899743b_left-54812_top-58969_bottom-59272_right-55081.csv'
    # display_image_with_box(image_path, 154, 270, 234, 350)
    visualize_image(image_path, file_path)
