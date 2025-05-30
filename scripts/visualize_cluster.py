import sys
sys.path.append('.')

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.utils import load_image
from src.cluster import ClusterHelper

img_dataset_path = Path("weizmann2Images")
save_visualization_path = Path("sandbox/results/cluster_visualization")
save_visualization_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for entry in tqdm(list(img_dataset_path.iterdir())):
        if entry.is_file() and entry.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img_np = load_image(entry.as_posix())

            initial_mask = ClusterHelper.get_initial_split(img_np, initial_num_clusters=2)
            fig = ClusterHelper.visualize_initial_split(img_np=img_np, initial_split=initial_mask)
            fig.savefig(save_visualization_path / f"{entry.stem}_initial_split.png")
            plt.close(fig)

