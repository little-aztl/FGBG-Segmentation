import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
from multiprocessing import Pool

from src.utils import load_image
from src.evaluation import load_gt, evaluate_segmentation
from src.cluster import ClusterHelper
from src.histogram import InitialWeightHelper
from src.graph import GraphHelper

image_dataset_path = Path("weizmann2Images")
gt_dataset_path = Path("weizmann2TruthOne")

def work(image_index, image_number):
    print(f"Processing No.{image_index + 1}: {image_number}")

    image_path = image_dataset_path / f"{image_number}.png"
    gt_path = gt_dataset_path / f"{image_number}_gt.png"
    img_color = load_image(image_path.as_posix())
    img_gray = load_image(image_path.as_posix(), mode='gray')
    initial_mask = ClusterHelper.get_initial_split(img_color, initial_num_clusters=2)
    initial_weight_helper = InitialWeightHelper(img_gray, initial_mask)
    foreground_weight, background_weight = initial_weight_helper.get_initial_weight()

    graph_helper = GraphHelper(img_color, foreground_weight, background_weight)
    seg_mask = graph_helper.get_segmentation_result().astype(bool)

    img_gt = load_gt(gt_path.as_posix())
    return evaluate_segmentation(seg_mask, img_gt)

if __name__ == "__main__":
    image_numbers = [entry.stem for entry in image_dataset_path.iterdir() if entry.is_file() and entry.suffix in ['.jpg', '.png', '.jpeg']]
    with Pool(processes=16) as pool:
        f1_scores = pool.starmap(work, enumerate(image_numbers))

    f1_scores = np.array(f1_scores)
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}")