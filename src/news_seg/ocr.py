from typing import Optional, List

import numpy as np
from PIL import Image
from PIL.Image import Resampling
from bs4 import BeautifulSoup
from kraken import blla, binarization, rpred
from kraken.lib import vgsl, models
from kraken.containers import Segmentation
from matplotlib import pyplot as plt
from skimage.draw import polygon
from tqdm import tqdm


def get_polygone(coords: str):
    polygone = np.array([tuple(map(int, point.split(','))) for point in
                     coords.split()])
    return polygone[:, ::-1]


def get_bbox(polygone: np.array):
    """
    input shape is ((N), P, 2)
    """
    if polygone.ndim == 2:
        polygone = polygone[None, :, :]

    x_max = np.amax(polygone[:, :, 0], axis=1)
    x_min = np.amin(polygone[:, :, 0], axis=1)
    y_max = np.amax(polygone[:, :, 1], axis=1)
    y_min = np.amin(polygone[:, :, 1], axis=1)

    return np.array([x_min, x_max, y_min, y_max]).flatten()


def extract_paragraphs(xml_path):
    polygones = []
    bboxes = []

    with open(xml_path, 'r', encoding='utf-8') as file:
        xml_data = file.read()

    # Parse the XML data
    soup = BeautifulSoup(xml_data, 'xml')

    # Find all TextLine elements
    regions = soup.find_all('TextRegion')

    # Extract Baseline points from each TextLine
    for i, region in enumerate(regions):
        poly = get_polygone(region.find('Coords')['points'])
        bbox = get_bbox(poly)

        polygones.append(poly)
        bboxes.append(bbox)

    return polygones, bboxes


def plot_baselines_image(image: Image, segmentation: Segmentation):
    baselines = [x.baseline for x in segmentation.lines]

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image, cmap='gray')

    for baseline in baselines:
        baseline = np.array(baseline)
        ax.plot(baseline[:, 0], baseline[:, 1], color='blue', linewidth=0.2)

    plt.savefig('../../data/baselineExampleTest.png', dpi=450)


class OCR:
    def __init__(self, baseline_model_path: Optional[str] = None, ocr_model_path: Optional[str] = None):
        self.scale = 0.2

        model_path = (baseline_model_path if baseline_model_path is not None
                      else '../../models/ubma_segmentation/ubma_segmentation.mlmodel')
        self.baseline_model = vgsl.TorchVGSLModel.load_model(model_path)

        model_path = (ocr_model_path if ocr_model_path is not None
                      else '../../models/german_newspapers_kraken/german_newspapers_kraken.mlmodel')
        self.ocr_model = models.load_any(model_path, device='cuda:0')

    def baselines(self, image: Image, paragraphs: List[np.array], bboxes: List[np.array]) -> Segmentation:
        shape = int(image.size[0] * self.scale), int(image.size[1] * self.scale)
        image = image.resize(shape, resample=Resampling.BICUBIC)
        image = np.array(image)

        lines = []

        # Create a mask for the polygon
        for paragraph, bbox in tqdm(zip(paragraphs, bboxes), total=len(paragraphs), desc='predict baselines'):
            bbox = (bbox * self.scale).astype(int)
            paragraph = (paragraph * self.scale).astype(int)

            mask = np.zeros_like(image, dtype=np.uint8)
            rr, cc = polygon(paragraph[:, 0], paragraph[:, 1], image.shape)
            mask[rr, cc] = 1
            masked_image = image * mask
            area = Image.fromarray(masked_image[bbox[0]: bbox[1], bbox[2]: bbox[3]])

            baseline_seg = blla.segment(binarization.nlbin(area),
                                        model=self.baseline_model,
                                        device='cpu')

            for baseline in baseline_seg.lines:
                baseline.baseline = ((np.array(baseline.baseline) + np.array([bbox[2], bbox[0]])) / self.scale).astype(int)
                baseline.boundary = ((np.array(baseline.boundary) + np.array([bbox[2], bbox[0]])) / self.scale).astype(int)
                lines.append(baseline)


        baseline_seg = Segmentation(type='baselines',
                                    imagename='../../data/images/testimage1.jpg',
                                    text_direction='horizontal-lr',
                                    script_detection=False,
                                    lines=lines,
                                    line_orders=[])

        return baseline_seg

    def ocr(self, image: Image, baselines: Segmentation) -> List[str]:
        pred_it = rpred.rpred(self.ocr_model, image, baselines)
        lines = [str(line) for line in tqdm(pred_it, desc='predicting text')]

        return lines


if __name__ == '__main__':
    ocr = OCR()

    image = Image.open('../../data/images/baselineExample.jpg')

    paragraphs, bboxes = extract_paragraphs('../../data/annotations/baselineExample.xml')

    baseline_seg = ocr.baselines(image, paragraphs, bboxes)
    plot_baselines_image(image, baseline_seg)
    lines = ocr.ocr(image, baseline_seg)
    print(lines[0])
