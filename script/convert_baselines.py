import glob

import numpy as np
import skimage.io
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from skimage.draw import line, polygon
from scipy.signal import convolve2d
from tqdm import tqdm

from src.news_seg.utils import get_bbox


def extract_baselines(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    baselines = []
    boxes = []

    # Parse the XML data
    soup = BeautifulSoup(data, 'xml')
    page = soup.find('Page')
    width, height = page['imageWidth'], page['imageHeight']

    # Find all TextLine elements
    text_lines = page.find_all('TextLine')

    # Extract Baseline points from each TextLine
    for text_line in text_lines:
        bbox = text_line.find('Coords')
        baseline = text_line.find('Baseline')
        if baseline:
            baseline = np.array([tuple(map(int, point.split(','))) for point in baseline['points'].split()])[:, ::-1]
            box = np.array([tuple(map(int, point.split(','))) for point in bbox['points'].split()])[:, ::-1]
            baselines.append(baseline)
            boxes.append(box)

    return baselines, boxes, width, height


def draw_shapes(baselines, boxes, width, height):
    # Create two numpy arrays of size defined by image_shape
    image = np.zeros((height, width, 4), dtype=np.uint8)

    top_kernel = np.array([[1], [1], [-1]])
    buttom_kernel = np.array([[-1], [1], [1]])

    # Draw lines on the image
    for line_coords in baselines:
        for i in range(len(line_coords)-1):
            rr, cc = line(line_coords[i][0], line_coords[i][1], line_coords[i+1][0], line_coords[i+1][1])
            image[rr, cc, 0] = 1

    # Draw boxes on the image
    for box_coords in tqdm(boxes, desc='calc lines'):
        temp = np.zeros((height, width), dtype=np.uint8)

        rr, cc = polygon(box_coords[:, 0], box_coords[:, 1])
        image[rr, cc, 1] = 1
        temp[rr, cc] = 1

        x_min, y_min, x_max, y_max = get_bbox(box_coords)

        toplines = convolve2d(temp[x_min-1:x_max+1, y_min-1:y_max+1], top_kernel, mode='same')
        buttomlines = convolve2d(temp[x_min-1:x_max+1, y_min-1:y_max+1], buttom_kernel, mode='same')

        image[x_min-1:x_max+1, y_min-1:y_max+1, 2] = np.maximum((toplines >= 2).astype(np.int8), image[x_min-1:x_max+1, y_min-1:y_max+1, 2])
        image[x_min-1:x_max+1, y_min-1:y_max+1, 3] = np.maximum((buttomlines >= 2).astype(np.int8), image[x_min-1:x_max+1, y_min-1:y_max+1, 3])

    return image


def plot_target(image, target, figsize=(20, 10), dpi=1000):
    fig, axes = plt.subplots(1, 4, figsize=figsize)  # Adjust figsize as needed

    # Plot the first image
    axes[0].imshow(image)
    axes[0].imshow(target[:, :, 0], cmap='gray', alpha=0.5)
    axes[0].set_title('baselines')

    # Plot the second image
    axes[1].imshow(image)
    axes[1].imshow(target[:, :, 1], cmap='gray', alpha=0.5)
    axes[1].set_title('boxes')

    axes[2].imshow(image)
    axes[2].imshow(target[:, :, 2], cmap='gray', alpha=0.5)
    axes[2].set_title('toplines')

    axes[3].imshow(image)
    axes[3].imshow(target[:, :, 3], cmap='gray', alpha=0.5)
    axes[3].set_title('buttomlines')

    # Turn off axis for cleaner display
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()  # Adjust layout
    plt.subplots_adjust(wspace=0.05)  # Adjust space between subplots

    # Display the plot with higher DPI
    plt.savefig('../data/baselineModelTargets.png', dpi=dpi)
    plt.show(dpi=dpi)


def main(xml_data):
    image = skimage.io.imread('../data/images/baselineExample.jpg')

    files = glob.glob(f"{xml_data}ba*.xml")
    print(f"{files=}")
    for file in files:
        baselines, boxes, width, height = extract_baselines(file)
        target = draw_shapes(baselines, boxes, int(width), int(height))
        break

    plot_target(image, target)


if __name__ == '__main__':
    main('../data/annotations/')
