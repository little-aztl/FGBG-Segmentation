import numpy as np
from PIL import Image

def load_image(image_path, mode='RGB'):
    '''
    Loads an image from the specified path and converts it to a NORMALIZED numpy array.

    RANGE: [0, 1]
    '''
    try:
        img = Image.open(image_path)
        if mode == 'gray':
            img = img.convert('L')
        img_np = np.array(img) / 255.0
        return img_np
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None