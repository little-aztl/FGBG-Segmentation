import sys
sys.path.append('.')

from src.utils import load_image
from src.cluster import ClusterHelper
from src.histogram import InitialWeightHelper
from src.graph import GraphHelper

from src.visualize import get_visualization

if __name__ == "__main__":
    test_image_path = 'weizmann2Images/2011.png'
    img_color = load_image(test_image_path, mode='RGB')
    img_gray = load_image(test_image_path, mode='gray')

    initial_mask = ClusterHelper.get_initial_split(img_color, initial_num_clusters=2)
    initial_weight_helper = InitialWeightHelper(img_gray, initial_mask)
    foreground_weight, background_weight = initial_weight_helper.get_initial_weight()

    graph_helper = GraphHelper(img_color, foreground_weight, background_weight)
    seg_mask = graph_helper.get_segmentation_result()

    fig = get_visualization(img_color, seg_mask)
    fig.savefig('sandbox/results/segmentation_result.png')