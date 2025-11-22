import matplotlib.pyplot as plt

def show_images(images, titles=None, cmaps=None, figsize=(16,4)):
    """
    Display multiple images in a single row.
    
    Parameters
    ----------
    images : list of np.ndarray
        List of images to display.
    titles : list of str, optional
        Titles for each image. Defaults to None.
    cmaps : list of str or None, optional
        Colormaps for each image. Defaults to None (auto).
    figsize : tuple, optional
        Figure size. Defaults to (16,4). mean four images each one of them size 4x4.
    """
    n = len(images)
    if titles is None:
        titles = ['']*n
    if cmaps is None:
        cmaps = [None]*n

    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], cmap=cmaps[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
