from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt

TRANSPARENT_TO_RED = LinearSegmentedColormap.from_list(
    "TransparentToRed", 
    [(1, 0, 0, 0), (1, 0, 0, 1)]
)

def plot_probabilities(rois, probabilities, title):
    prob_rois = (rois>0) * probabilities[:, 1, np.newaxis, np.newaxis]
    fig, ax = plt.subplots()
    ax.imshow(rois.sum(axis=0), cmap='gray')
    im = ax.imshow(prob_rois.sum(axis=0), cmap=TRANSPARENT_TO_RED, clim=[0,1], alpha=.9)
    ax.set_title(title)
    return ax

def plot_predictions(rois, predictions, title):
    pred_rois = (rois>0) * predictions[:, np.newaxis, np.newaxis]
    fig, ax = plt.subplots()
    ax.imshow(rois.sum(axis=0), cmap='gray')
    ax.imshow(pred_rois.sum(axis=0), cmap=TRANSPARENT_TO_RED, alpha=.9)
    ax.set_title(title)
    return ax