import sys
sys.path.append('.')

from src.utils import load_image

if __name__ == "__main__":
    test_image_path = 'weizmann2Images/2004.png'
    img_np = load_image(test_image_path, mode='gray')
    print(img_np.dtype, img_np.shape)