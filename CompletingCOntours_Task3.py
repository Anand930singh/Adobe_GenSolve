import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage.measure import EllipseModel, CircleModel
from skimage.draw import ellipse_perimeter, circle_perimeter
from scipy.spatial import ConvexHull
from skimage.transform import hough_circle, hough_circle_peaks

# Function to read CSV and add headers
def read_csv_with_headers(file_name):
    df = pd.read_csv(file_name, header=None)
    df.columns = ['path_id', 'segment_id', 'x-coordinate', 'y-coordinate']
    # Invert the y-coordinates if necessary
    max_y = df['y-coordinate'].max()
    df['y-coordinate'] = max_y - df['y-coordinate']
    return df

# Load the CSV file
df = read_csv_with_headers('occlusion1.csv')

# Function to get contours from the data
def get_contours(df):
    contours = []
    unique_paths = df['path_id'].unique()
    for path_id in unique_paths:
        path_data = df[df['path_id'] == path_id]
        contour = path_data[['x-coordinate', 'y-coordinate']].values
        contours.append((path_id, contour))
    return contours

# Get contours from the dataset
contours = get_contours(df)

# Function to plot contours
def plot_contours(contours, save_images=False):
    for path_id, contour in contours:
        plt.plot(contour[:, 0], contour[:, 1])
        if save_images:
            plt.savefig(f'{path_id}_shape.png')  # Save in the current directory
        plt.clf()

# Separate overlapping shapes using DBSCAN
def separate_overlapping_shapes(contours):
    separated_shapes = []
    for path_id, contour in contours:
        clustering = DBSCAN(eps=10, min_samples=2).fit(contour)
        labels = clustering.labels_
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            shape = contour[labels == label]
            separated_shapes.append((path_id, shape))
    return separated_shapes

# Separate overlapping shapes
separated_shapes = separate_overlapping_shapes(contours)

# Function to fit and regularize shapes
def fit_and_regularize_shapes(separated_shapes):
    regularized_shapes = []
    for path_id, shape in separated_shapes:
        # Fit to circle using Hough Transform
        min_x, min_y = np.min(shape, axis=0)
        max_x, max_y = np.max(shape, axis=0)
        image = np.zeros((int(max_y - min_y) + 1, int(max_x - min_x) + 1), dtype=np.uint8)
        shape_translated = shape - [min_x, min_y]
        rr, cc = shape_translated[:, 1].astype(int), shape_translated[:, 0].astype(int)
        image[rr, cc] = 1

        hough_radii = np.arange(5, 50, 1)
        hough_res = hough_circle(image, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, threshold=0.5)
        
        detected_circle = None
        if len(cx) > 0 and len(radii) > 0:  # Check if any circles are detected
            for center_x, center_y, radius in zip(cx, cy, radii):
                if radius < 10:  # Filter based on radius
                    detected_circle = (center_x + min_x, center_y + min_y, radius)
                    break

        if detected_circle:
            cx, cy, r = detected_circle
            rr, cc = circle_perimeter(int(cy), int(cx), int(r))
            rr, cc = np.clip(rr, 0, image.shape[0] - 1), np.clip(cc, 0, image.shape[1] - 1)
            regularized_shape = np.column_stack((cc, rr)).astype(float)
            shape_name = 'circle'
        else:
            # Fit to ellipse
            ellipse = EllipseModel()
            ellipse_fit = ellipse.estimate(shape)
            
            # Fit to rectangle (using bounding box)
            min_x, min_y = np.min(shape, axis=0)
            max_x, max_y = np.max(shape, axis=0)
            rectangle_fit = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
            
            if ellipse_fit:
                cy, cx, a, b, orientation = ellipse.params
                rr, cc = ellipse_perimeter(int(cy), int(cx), int(a), int(b), orientation)
                regularized_shape = np.column_stack((cc, rr)).astype(float)
                shape_name = 'ellipse'
            elif len(shape) >= 4:  # Heuristic: use rectangle if shape has at least 4 points
                regularized_shape = rectangle_fit.astype(float)
                shape_name = 'rectangle'
            else:
                hull = ConvexHull(shape)
                regularized_shape = shape[hull.vertices].astype(float)
                shape_name = 'polygon'
        
        # Translate the regularized shape back to its original position
        translation = shape.mean(axis=0) - regularized_shape.mean(axis=0)
        regularized_shape += translation
        
        regularized_shapes.append((path_id, shape_name, regularized_shape.astype(int)))
    return regularized_shapes

# Regularize shapes
regularized_shapes = fit_and_regularize_shapes(separated_shapes)

# Function to plot and save regularized shapes at their original positions
def plot_and_save_regularized_shapes(original_shapes, regularized_shapes):
    for (path_id, original_shape), (_, shape_name, regularized_shape) in zip(original_shapes, regularized_shapes):
        plt.plot(original_shape[:, 0], original_shape[:, 1], label='Original')
        plt.plot(regularized_shape[:, 0], regularized_shape[:, 1], label=f'Regularized ({shape_name})')
        plt.legend()
        plt.title(f'Shape ID: {path_id} - {shape_name}')
        plt.savefig(f'{path_id}_{shape_name}.png')  # Save in the current directory
        plt.show()
        plt.clf()

# Plot and save the regularized shapes
plot_and_save_regularized_shapes(contours, regularized_shapes)