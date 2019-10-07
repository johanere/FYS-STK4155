import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
def read_terrain(terrain):
    terrain1 = imread(terrain)
    return terrain1


# Show the terrain


def plot_terrain(terrain):
    plt.figure()
    plt.title("Terrain over Norway 1")
    plt.imshow(terrain, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
