# Adobe GenSolve - Innovate to Impact

## Shape Detection using OpenCV

This project is designed to detect and categorize geometric shapes (such as circles, rectangles, squares, triangles, and stars) in an image using OpenCV. It works by identifying the contours of objects in an image, approximating those contours to polygons, and then analyzing their properties to determine the type of shape.

### Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Functionality](#functionality)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [License](#license)

### Requirements

- Python 3.x
- OpenCV
- NumPy

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/shape-detection.git
   cd shape-detection

2. # Curvetopia: A Journey into the World of Curves

Welcome to Curvetopia! This project explores the identification, regularization, and beautification of various 2D curves. We focus on developing techniques to handle closed curves, symmetry, and curve completion.

## Objective

Our mission is to create an end-to-end process that takes a line art input in the form of polylines and outputs a set of curves defined as connected sequences of cubic Bézier curves. The primary goals include:

- **Curve Regularization:** Identifying and regularizing basic geometric shapes such as lines, circles, ellipses, rectangles, and polygons.
- **Symmetry Exploration:** Detecting symmetries in closed curves and representing them efficiently.
- **Curve Completion:** Completing incomplete curves that may have gaps due to occlusions or other reasons.

## Problem Description

We simplify the input by using polylines instead of raster images. The input is a finite subset of paths, where each path is a sequence of points. The output is another set of paths with properties of regularization, symmetry, and completeness.

For visualization, the output curve will be in the form of cubic Bézier curves in an SVG format, allowing browser rendering.

## Sections

### 1. Regularize Curves

Identify regular shapes from a set of curves. The task can be broken down into identifying:

- Circles and Ellipses
- Rectangles and Rounded Rectangles
- Regular Polygons
- Star Shapes

### Steps we followed for regularization
1. Image Processing
  Loading Image and Converting Image to grayscale.
  We have used Canny Edge detection for identifying edges in the shape.
  The contours in the image in the image are found.
2. Shape Grouping and Detection
  The code groups similar shapes based on contour area, number of edges, and centroid proximity. This done to eliminate the multiple edge of same shape due to inside and outside boundary.
  Approximates the contour to a polygon using approxPolyDP.
  Identifies the type of shape (e.g., circle, triangle, square, rectangle, star) based on the number of edges and geometric properties.
  Draws the identified shapes on a blank image (shape_image).
  Prints the shape's details, such as the type, number of edges, and area, to the terminal.
4. Displaying the Results
  The resultant image with detected shapes is resized based on the specified resize factor.
  The resized image is displayed in a window, showing the detected shapes and their details.

### 2. Exploring Symmetry in Curves

Identify reflection symmetries in closed shapes. The task involves transforming the curve representation into points and fitting Bézier curves to the symmetric points.

### 3. Completing Incomplete Curves

Develop algorithms to naturally complete incomplete 2D curves that have gaps due to occlusions. The challenge is to determine how to complete these curves by analyzing smoothness, regularity, and symmetry.

### Steps we followed for completion of occludded shapes
1. We are extracting contours from the DataFrame based on path_id. Each contour represents a series of x and y coordinates for a specific shape.
2. We are using DBSCAN clustering to separate overlapping shapes into distinct groups. It returns a list of separated shapes.
3.Then we are trying fit each separated shape into a standard geometric form:
Circle: Using the Hough Transform.
Ellipse: Using an ellipse fitting model.
Rectangle: By calculating the bounding box.
Polygon: For more complex shapes.
4.The function translates each regularized shape back to its original position and stores the results.

