from matplotlib import pyplot as plt

def get_visualization(ori_img, seg_img):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(ori_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(seg_img, cmap='gray')
    axs[1].set_title('Segmented Image')
    axs[1].axis('off')

    return fig