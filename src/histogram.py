import numpy as np

class InitialWeightHelper:
    BINS = 8
    def __init__(self, img_gray, initial_mask):
        '''
        img_gray: (H, W)
        initial_mask: (H, W)
        '''
        self.img = img_gray
        self.img_flatten = img_gray.reshape(-1) # (H*W,)
        self.initial_mask_flatten = initial_mask.reshape(-1) # (H*W,)
        self._compute_histogram()

    def _compute_histogram(self):
        foreground_colors = self.img_flatten[self.initial_mask_flatten == 1]
        background_colors = self.img_flatten[self.initial_mask_flatten == 0]

        self.foreground_hist, self.foreground_edges = np.histogram(foreground_colors, bins=InitialWeightHelper.BINS, range=(0, 1))
        self.foreground_probs = (self.foreground_hist + 1) / (foreground_colors.size + InitialWeightHelper.BINS)  # Laplace smoothing
        self.background_hist, self.background_edges = np.histogram(background_colors, bins=InitialWeightHelper.BINS, range=(0, 1))
        self.background_probs = (self.background_hist + 1) / (background_colors.size + InitialWeightHelper.BINS)

    def get_initial_weight(self):
        foreground_bin_index = np.digitize(self.img, bins=self.foreground_edges)
        background_weight = -np.log(self.foreground_probs[foreground_bin_index - 1])

        background_bin_index = np.digitize(self.img, bins=self.background_edges)
        foreground_weight = -np.log(self.background_probs[background_bin_index - 1])

        return foreground_weight, background_weight