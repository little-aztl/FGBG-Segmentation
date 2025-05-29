import sys
sys.path.append('.')

from src.utils import load_image
from src.cluster import ClusterHelper
from src.histogram import InitialWeightHelper

if __name__ == "__main__":
    test_image_path = 'weizmann2Images/2004.png'
    img_color = load_image(test_image_path, mode='RGB')
    img_gray = load_image(test_image_path, mode='gray')

    initial_mask = ClusterHelper.get_initial_split(img_color, initial_num_clusters=2)
    initial_weight_helper = InitialWeightHelper(img_gray, initial_mask)
    initial_weight_helper.get_initial_weight()