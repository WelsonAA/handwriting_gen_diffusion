import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def remove_whitespace(img, thresh, remove_middle=True):
    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)
    
    rows = np.where(row_mins < thresh)
    cols = np.where(col_mins < thresh)

    if remove_middle:
        return img[rows[0]][:, cols[0]]
    else:
        rows, cols = rows[0], cols[0]
        return img[rows[0]:rows[-1], cols[0]:cols[-1]]

def read_img_debug(height, save_path="debug_output.png", path=None):
    """
    Loads an image from `path`, removes whitespace, resizes it to the given height,
    and saves a before/after comparison to `save_path`, using a dark background.
    """
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    if path is None:
        raise ValueError("You must specify the image `path`.")

    # Load original image
    img = load_img(path, color_mode='grayscale')
    img_arr = img_to_array(img).astype('uint8')

    # Remove whitespace
    img_arr_clean = remove_whitespace(img_arr, thresh=127)

    # Create figure with dark background
    plt.figure(figsize=(10, 4), facecolor='black')
    plt.subplot(1, 2, 1, facecolor='black')
    plt.imshow(img_arr.squeeze(), cmap='gray_r')
    plt.title('Original Image', color='white')
    plt.axis('off')

    plt.subplot(1, 2, 2, facecolor='black')
    plt.imshow(img_arr_clean.squeeze(), cmap='gray_r')
    plt.title('After Whitespace Removal', color='white')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, facecolor='black')
    plt.close()

    # Resize cleaned image
    h, w = img_arr_clean.shape[:2]
    img_arr_resized = tf.image.resize(img_arr_clean, (height, height * w // h))
    return img_arr_resized.numpy().astype('uint8')

# ==== Main block ====
if __name__ == "__main__":
    # Replace with your actual .tif image path
    tif_path = "data/lineImages-all/a01/a01-000/a01-000u-01.tif"
    read_img_debug(height=96, save_path="comparison.png", path=tif_path)
    print("Comparison saved to comparison.png")
