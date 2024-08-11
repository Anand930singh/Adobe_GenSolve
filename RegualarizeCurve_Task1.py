import cv2
import numpy as np

# Configuration
config = {
    "image_path": "./PNG/isolated.png", #image path
    "slope_tolerance": 0.1,  # Maximum slope deviation for grouping edges
    "area_threshold": 600,  # Minimum contour area to be considered (adjust as needed)
    "area_similarity_tolerance": 50000,   # Tolerance for considering areas similar
    "centroid_similarity_tolerance": 10,  # Tolerance for considering centroids similar in pixels
    "resize_factor": 0.5,  # Factor to resize the image for display
    "edge_length_tolerance": 0.2  # Tolerance for equal edge lengths in stars
}

def calculate_slope(p1, p2):
    """Calculate the slope of the line segment between points p1 and p2."""
    if p1[0] == p2[0]:  # Avoid division by zero
        return np.inf
    return (p2[1] - p1[1]) / (p1[0] - p2[0])

def are_similar_slopes(slope1, slope2, tolerance):
    """Check if two slopes are similar within a tolerance."""
    return abs(slope1 - slope2) < tolerance

def group_edges(approx, slope_tolerance):
    """Group adjacent edges with similar slopes."""
    edges = []
    num_points = len(approx)
    
    for i in range(num_points):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % num_points][0]
        slope = calculate_slope(p1, p2)
        
        if not edges:
            edges.append((slope, [p1, p2]))
        else:
            add_to_group = True
            for edge_slope, _ in edges:
                if not are_similar_slopes(slope, edge_slope, slope_tolerance):
                    add_to_group = False
                    break
            
            if add_to_group:
                last_slope, last_edge = edges[-1]
                last_edge.append(p2)
            else:
                edges.append((slope, [p1, p2]))
    
    return edges

def draw_edges(image, edges):
    """Draw edges on the image."""
    for slope, edge in edges:
        for i in range(len(edge) - 1):
            cv2.line(image, tuple(edge[i]), tuple(edge[i + 1]), (0, 255, 0), 2)

def draw_circle(image, contour):
    """Fit a circle to the contour and draw it on the image."""
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(image, center, radius, (255, 0, 0), 2)

def draw_star(image, approx):
    """Draw the star shape based on polygon approximation."""
    cv2.drawContours(image, [approx], 0, (0, 255, 255), 2)

def draw_polygon(image, approx, color=(0, 255, 0)):
    """Draw a polygon shape based on polygon approximation."""
    cv2.drawContours(image, [approx], 0, color, 2)

def is_rectangle_or_square(approx):
    """Check if the contour is a rectangle or square."""
    if len(approx) != 4:
        return False, False
    
    (x, y, w, h) = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h
    square = 0.95 <= aspect_ratio <= 1.05
    rectangle = not square
    return rectangle, square

def is_circle(contour, tolerance=0.2):
    """Check if the contour is approximately a circle."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return False
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity > (1 - tolerance)

def is_star(approx, tolerance=0.2):
    """Check if the contour is approximately a star with equal edge lengths."""
    if len(approx) != 10:
        return False

    edge_lengths = []
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        edge_length = np.linalg.norm(p1 - p2)
        edge_lengths.append(edge_length)
    
    mean_length = np.mean(edge_lengths)
    for length in edge_lengths:
        if abs(length - mean_length) > tolerance * mean_length:
            return False
    
    return True

def print_shape_info(name, approx, num_edges, area):
    """Print the shape information to the terminal including area."""
    print(f'{name} detected with {num_edges} edges, area {area:.2f} at points: {approx[:, 0].tolist()}')

def get_centroid(contour):
    """Calculate the centroid of a contour."""
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return (0, 0)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return (cx, cy)

# Load image and convert to grayscale
image = cv2.imread(config["image_path"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image for drawing shapes
shape_image = np.zeros_like(image)

# Group similar shapes
groups = []

for contour in contours:
    # Check if the contour's area is within the allowed range
    area = cv2.contourArea(contour)
    if area < config["area_threshold"]:
        continue

    # Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    centroid = get_centroid(contour)
    num_edges = len(approx)
    
    matched = False
    for group in groups:
        group_centroid, group_area, group_num_edges, group_contours = group
        if (abs(group_area - area) / group_area < config["area_similarity_tolerance"] and
            np.linalg.norm(np.array(group_centroid) - np.array(centroid)) < config["centroid_similarity_tolerance"] and
            group_num_edges == num_edges):
            group_contours.append(contour)
            matched = True
            break
    
    if not matched:
        groups.append((centroid, area, num_edges, [contour]))

# Draw shapes with the smallest areas
for centroid, _, num_edges, group_contours in groups:
    smallest_contour = min(group_contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(smallest_contour, 0.02 * cv2.arcLength(smallest_contour, True), True)
    area = cv2.contourArea(smallest_contour)

    # Determine shape type
    if is_circle(smallest_contour):
        draw_circle(shape_image, smallest_contour)
        print_shape_info('Circle', approx, num_edges, area)
    elif len(approx) == 3:
        draw_polygon(shape_image, approx, color=(0, 255, 0))  # Green for triangles
        print_shape_info('Triangle', approx, num_edges, area)
    elif len(approx) == 4:
        rectangle, square = is_rectangle_or_square(approx)
        if square:
            draw_polygon(shape_image, approx, color=(0, 255, 255))  # Cyan for squares
            print_shape_info('Square', approx, num_edges, area)
        elif rectangle:
            draw_polygon(shape_image, approx, color=(255, 0, 0))  # Red for rectangles
            print_shape_info('Rectangle', approx, num_edges, area)
    elif is_star(approx, config["edge_length_tolerance"]):
        draw_polygon(shape_image, approx, color=(0, 0, 255))  # Blue for stars
        print_shape_info('Star', approx, num_edges, area)
    else:
        cv2.drawContours(shape_image, [smallest_contour], 0, (255, 255, 255), 2)  # White for other shapes
        print_shape_info('Contour', approx, num_edges, area)

    # Display the number of edges
    cv2.putText(shape_image, f'Edges: {num_edges}', (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Resize the image for a smaller display window
width = int(image.shape[1] * config["resize_factor"])
height = int(image.shape[0] * config["resize_factor"])
dim = (width, height)
resized_image = cv2.resize(shape_image, dim, interpolation=cv2.INTER_AREA)

# Show the result
cv2.imshow('Detected Shapes', resized_image)            
cv2.waitKey(0)
cv2.destroyAllWindows()
