from PIL import Image
import numpy as np


def export_to_jpg(matrix, filename):
    # Normalize the matrix to the range [0, 255]
    normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255
    normalized_matrix = normalized_matrix.astype(np.uint8)

    # Create an image from the normalized matrix
    img = Image.fromarray(normalized_matrix)

    # Save the image as a JPEG file
    img.save(filename)

def export_sim_to_jpg(sim, time, folder="datasets/RT/", prefix="sim"):
    for i in range(sim.shape[0]):
        filename = f"{folder}{prefix}_{i}.jpg"
        export_to_jpg(sim[i], filename)
        # print(f"Exported {filename} for time={time[i]:.2f}")