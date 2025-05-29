import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten
from matplotlib import pyplot as plt

class ClusterHelper(object):
    def _get_pixel_data(img_np):
        pixel_data = np.zeros((img_np.shape[0] * img_np.shape[1], 5))

        pixel_data[:, 2:] = img_np.reshape(-1, 3)
        X, Y = np.meshgrid(
            np.arange(img_np.shape[0]), np.arange(img_np.shape[1]), indexing='ij'
        )
        pixel_data[:, 0] = X.reshape(-1) / img_np.shape[0]
        pixel_data[:, 1] = Y.reshape(-1) / img_np.shape[1]
        pixel_data[:, :2] *= 0.1
        return pixel_data

    def _cluster_pixels(pixel_data, num_clusters):
        centroids, distortion = kmeans(pixel_data, num_clusters, iter=20)

        cluster_ids, _ = vq(pixel_data, centroids)
        return cluster_ids


    @staticmethod
    def get_initial_split(img_np, initial_num_clusters):
        pixel_data = ClusterHelper._get_pixel_data(img_np)
        cluster_ids = ClusterHelper._cluster_pixels(pixel_data, initial_num_clusters)

        max_cluster_id, max_cluster_count = -1, -1
        for cluster_id in range(initial_num_clusters):
            cluster_count = np.sum(cluster_ids == cluster_id)
            if cluster_count > max_cluster_count:
                max_cluster_id, max_cluster_count = cluster_id, cluster_count

        initial_split = np.zeros_like(cluster_ids)
        initial_split[cluster_ids == max_cluster_id] = 0
        initial_split[cluster_ids != max_cluster_id] = 1
        return initial_split.reshape(img_np.shape[0], img_np.shape[1])

    @staticmethod
    def visualize_clusters(img_np, cluster_ids, num_clusters):
        colors = np.array([
            [1.0, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        cluster_image = np.zeros_like(img_np).reshape(-1, 3)
        for cluster_id in range(num_clusters):
            cluster_mask = (cluster_ids == cluster_id)
            cluster_image[cluster_mask] = colors[cluster_id % len(colors)]

        cluster_image = cluster_image.reshape(img_np.shape)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img_np)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(img_np)
        axs[1].imshow(cluster_image, alpha=0.5)
        axs[1].set_title('Clustered Image')
        axs[1].axis('off')

        return fig

    @staticmethod
    def visualize_initial_split(img_np, initial_split):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img_np)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(initial_split, cmap='gray')
        axs[1].set_title('Initial Split')
        axs[1].axis('off')
        return fig




