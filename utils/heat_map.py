import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


class HeatMap:
    def __init__(self, image, heat_map, heat_map_gt, gaussian_std=10):
        # if image is numpy array
        if isinstance(image, np.ndarray):
            height = image.shape[0]
            width = image.shape[1]
            self.image = image
        else:
            # PIL open the image path, record the height and width
            image = Image.open(image)
            width, height = image.size
            self.image = image

        # Convert numpy heat_map values into image formate for easy upscale
        # Rezie the heat_map to the size of the input image
        # Apply the gausian filter for smoothing
        # Convert back to numpy
        if np.ndim(heat_map) > 2:
            heat_map = np.squeeze(heat_map, axis=-1)
        if np.ndim(heat_map_gt) > 2:
            heat_map_gt = np.squeeze(heat_map_gt, axis=-1)

        heatmap_image = Image.fromarray(heat_map * 255)
        heatmap_image_resized = heatmap_image.resize((width, height))

        heatmap_gt_image = Image.fromarray(heat_map_gt * 255)
        heatmap_gt_image_resized = heatmap_gt_image.resize((width, height))
        if gaussian_std > 0:
            heatmap_image_resized = ndimage.gaussian_filter(heatmap_image_resized,
                                                            sigma=(gaussian_std, gaussian_std),
                                                            order=0)
            heatmap_gt_image_resized = ndimage.gaussian_filter(heatmap_gt_image_resized,
                                                            sigma=(gaussian_std, gaussian_std),
                                                            order=0)

        heatmap_image_resized = np.asarray(heatmap_image_resized)
        heatmap_gt_image_resized = np.asarray(heatmap_gt_image_resized)
        self.heat_map = heatmap_image_resized
        self.heat_map_gt = heatmap_gt_image_resized

    # Plot the figure
    def plot(
            self,
            transparency=0.5,
            color_map='jet',
            show_axis=False,
            show_original=False,
            show_colorbar=False,
            width_pad=0,
            figure_size=(10, 20),
            text=None,
            text_fontsize='large',
            text_fontweight='bold',
            text_color='white'
    ):

        # If show_original is True, then subplot first figure as orginal image
        # Set x,y to let the heatmap plot in the second subfigure,
        # otherwise heatmap will plot in the first sub figure
        plt.figure(figsize=figure_size)
        axes_title_fontdict = {
            'fontsize': text_fontsize,
            'fontweight': text_fontweight,
            'color': text_color
        }

        if show_original:
            axes_org_img = plt.subplot(1, 3, 1)
            axes_org_img.set_title('Image', fontdict=axes_title_fontdict)
            if not show_axis:
                plt.axis('off')
            plt.imshow(self.image)
            x, y = 2, 2
        else:
            x, y = 1, 1

        # Plot the ground truth
        axes_predict_hm = plt.subplot(1, 3, 2)
        axes_predict_hm.set_title('Ground Truth', fontdict=axes_title_fontdict)
        if not show_axis:
            plt.axis('off')
        plt.imshow(self.image, cmap='gray')
        plt.imshow(self.heat_map_gt / 255, alpha=transparency, cmap=color_map)

        # Plot the heatmap
        axes_predict_hm = plt.subplot(1, 3, 3)
        axes_predict_hm.set_title('Predicted: {}'.format(text), fontdict=axes_title_fontdict)
        if not show_axis:
            plt.axis('off')
        plt.imshow(self.image, cmap='gray')
        plt.imshow(self.heat_map / 255, alpha=transparency, cmap=color_map)

        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.show()

    ###Save the figure
    def save(self, filename, format='png', save_path=os.getcwd(),
             transparency=0.7, color_map='bwr', width_pad=-10,
             show_axis=False, show_original=False, show_colorbar=False, **kwargs):
        if show_original:
            axes_org_img = plt.subplot(1, 2, 1)
            axes_org_img.set_title('Image')
            if not show_axis:
                plt.axis('off')
            plt.imshow(self.image)
            x, y = 2, 2
        else:
            x, y = 1, 1

        # Plot the heatmap
        axes_predict_hm = plt.subplot(1, x, y)
        axes_predict_hm.set_title('Predicted')
        if not show_axis:
            plt.axis('off')
        plt.imshow(self.image)
        plt.imshow(self.heat_map / 255, alpha=transparency, cmap=color_map)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.savefig(os.path.join(save_path, filename + '.' + format),
                    format=format,
                    bbox_inches='tight',
                    pad_inches=0, **kwargs)
        print('{}.{} has been successfully saved to {}'.format(filename, format, save_path))
