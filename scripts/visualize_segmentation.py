import sys
sys.path.append('.')

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from multiprocessing import Pool

from src.utils import load_image
from src.cluster import ClusterHelper
from src.histogram import InitialWeightHelper
from src.graph import GraphHelper

from src.visualize import get_visualization

image_dataset_path = Path("weizmann2Images")
save_visualization_path = Path("sandbox/results/segmentation_visualization")
save_visualization_path.mkdir(parents=True, exist_ok=True)

def work(entry_index, image_entry):
    print(f"Processing {entry_index + 1}: {image_entry.name}")
    img_color = load_image(image_entry.as_posix())
    img_gray = load_image(image_entry.as_posix(), mode='gray')
    initial_mask = ClusterHelper.get_initial_split(img_color, initial_num_clusters=2)
    initial_weight_helper = InitialWeightHelper(img_gray, initial_mask)
    foreground_weight, background_weight = initial_weight_helper.get_initial_weight()

    graph_helper = GraphHelper(img_color, foreground_weight, background_weight)
    seg_mask = graph_helper.get_segmentation_result()

    fig = get_visualization(img_color, seg_mask)
    fig.savefig((save_visualization_path / f"{image_entry.stem}_segmentation.png").as_posix())
    plt.close(fig)

if __name__ == "__main__":
    entries = [entry for entry in image_dataset_path.iterdir() if entry.is_file() and entry.suffix in ['.jpg', '.png', '.jpeg']]
    with Pool(processes=16) as pool:
        pool.starmap(work, enumerate(entries))

