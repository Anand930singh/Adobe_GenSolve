# Adobe GenSolve - Innovate to Impact

## Shape Detection using OpenCV

This project is designed to detect and categorize geometric shapes (such as circles, rectangles, squares, triangles, and stars) in an image using OpenCV. It works by identifying the contours of objects in an image, approximating those contours to polygons, and then analyzing their properties to determine the type of shape.

### Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Objective](#objective)
- [Problem Description](#problem-description)
- [Reading CSV](#reading-csv)
- [Regularize Curves](#regularize-curves)
- [Exploring Symmetry in Curves](#exploring-symmetry-in-curves)
- [Completing Incomplete Curves](#completing-incomplete-curves)

### Requirements

- Python 3.x
- OpenCV
- NumPy
- BytesIO
- cairosvg

### Running the Code

To make it easier to run the code without installing any libraries or setting up the environment, we have created a Google Colab notebook. You can run the code directly in the Colab environment by following the link below:

- [Google Colab Notebook](https://colab.research.google.com/drive/1WTm6OUnrqQ5u9VvU0v8Z8_Aiw2daF06F?usp=sharing)

This notebook includes step-by-step instructions and allows you to execute the code in the cloud, making it convenient to test and experiment with the shape detection algorithms without any local setup.


### Installation

**Clone the repository:**

   ```bash
   git clone https://github.com/your-username/shape-detection.git
   cd shape-detection
```



## Objective

Our mission is to create an end-to-end process that takes a line art input in the form of polylines and outputs a set of curves defined as connected sequences of cubic Bézier curves. The primary goals include:

- **Curve Regularization:** Identifying and regularizing basic geometric shapes such as lines, circles, ellipses, rectangles, and polygons.
- **Symmetry Exploration:** Detecting symmetries in closed curves and representing them efficiently.
- **Curve Completion:** Completing incomplete curves that may have gaps due to occlusions or other reasons.

## Problem Description

We simplify the input by using polylines instead of raster images. The input is a finite subset of paths, where each path is a sequence of points. The output is another set of paths with properties of regularization, symmetry, and completeness.

For visualization, the output curve will be in the form of cubic Bézier curves in an SVG format, allowing browser rendering.


### Reading CSV
1. First wer are generating PNG and SVG image from the provided CSV files for applying the methods of openCV.

### Regularize Curves

Identify regular shapes from a set of curves. The task can be broken down into identifying:

- Circles and Ellipses
- Rectangles and Rounded Rectangles
- Regular Polygons
- Star Shapes

## Steps we followed for regularization
1. Image Processing
  - Loading Image and Converting Image to grayscale.
  - Used Canny Edge detection for identifying edges in the shape.
  - The contours in the image in the image are found.
    
3. Shape Grouping and Detection
  - The code groups similar shapes based on contour area, number of edges, and centroid proximity. This done to eliminate the multiple edge of same shape due to inside and outside boundary.
  - Approximates the contour to a polygon using approxPolyDP.
  - Identifies the type of shape (e.g., circle, triangle, square, rectangle, star) based on the number of edges and geometric properties.
  - Draws the identified shapes on a blank image (shape_image).
  - Prints the shape's details, such as the type, number of edges, and area, to the terminal.
    
4. Displaying the Results
  - The resultant image with detected shapes is resized based on the specified resize factor.
  - The resized image is displayed in a window, showing the detected shapes and their details.
<img width="866" alt="image" src="https://github.com/user-attachments/assets/a5dae527-7e3f-43ff-9ae9-377577a1b8ef">


### Exploring Symmetry in Curves

- Identify reflection symmetries in closed shapes. The task involves transforming the curve representation into points and fitting Bézier curves to the symmetric points.

<img width="687" alt="image" src="https://github.com/user-attachments/assets/23ff5647-4aa7-43d8-9411-05525209fb7c">


### Completing Incomplete Curves

- Develop algorithms to naturally complete incomplete 2D curves that have gaps due to occlusions. The challenge is to determine how to complete these curves by analyzing smoothness, regularity, and symmetry.

### Steps we followed for completion of occluded shapes
- We are extracting contours from the DataFrame based on path_id. Each contour represents a series of x and y coordinates for a specific shape.
-  We are using DBSCAN clustering to separate overlapping shapes into distinct groups. It returns a list of separated shapes.
- Then we are trying fit each separated shape into a standard geometric form:
   Circle: Using the Hough Transform.
   Ellipse: Using an ellipse fitting model.
   Rectangle: By calculating the bounding box.
   Polygon: For more complex shapes.
- The function translates each regularized shape back to its original position and stores the results.

<img width="297" alt="image" src="https://github.com/user-attachments/assets/4e3af8ab-6c47-4736-bf73-b27d3b3503a1">

