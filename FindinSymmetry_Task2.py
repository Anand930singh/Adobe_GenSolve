import cv2
import numpy as np

# Configuration
config = {
    "image_path": "./png_svg/isolated.png",  # image path
    "slope_tolerance": 0.1,  # Maximum slope deviation for grouping edges
    "area_threshold": 600,  # Minimum contour area to be considered (adjust as needed)
    "area_similarity_tolerance": 0.1,  # Tolerance ratio for considering areas similar
    "centroid_similarity_tolerance": 10,  # Tolerance for considering centroids similar in pixels
    "resize_factor": 0.5,  # Factor to resize the image for display
    "edge_length_tolerance": 0.2,  # Tolerance for equal edge lengths in stars
    "dotted_line_length": 10,  # Length of each dot in the dotted line
    "dotted_line_gap": 5  # Gap between dots in the dotted line
}

# (Previous functions remain unchanged)

def draw_dotted_line(img, pt1, pt2, color, thickness=1, line_length=10, gap=5):
    """Draw a dotted line between two points."""
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    pts = []
    for i in np.arange(0, dist, line_length + gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness)

def is_reflective_symmetry(contour, line):
    """Check if the contour is symmetric with respect to the given line."""
    pts = np.array(contour).reshape(-1, 2)
    line_params = np.polyfit(line[0], line[1], 1)  # Get line parameters

    def reflect_point(pt):
        """Reflect a point across the line."""
        x, y = pt
        slope, intercept = line_params
        if slope == np.inf:
            reflected_x = 2 * intercept - x
            return (reflected_x, y)
        reflected_x = (x - 2 * slope * (y - intercept)) / (1 + slope ** 2)
        reflected_y = slope * reflected_x + intercept
        return (reflected_x, reflected_y)

    reflected_pts = [reflect_point(pt) for pt in pts]
    reflected_pts = np.array(reflected_pts).astype(int)

    # Check if the reflected points are close to the original contour points
    distances = [cv2.pointPolygonTest(contour, (x, y), True) for (x, y) in reflected_pts]
    return all(d <= 1 for d in distances)  # Check if all reflected points are close to original contour


def is_reflective_symmetry(contour, line):
    """Check if the contour is symmetric with respect to the given line."""
    pts = np.array(contour).reshape(-1, 2)
    line_params = np.polyfit(line[0], line[1], 1)  # Get line parameters

    def reflect_point(pt):
        """Reflect a point across the line."""
        x, y = pt
        slope, intercept = line_params
        if slope == np.inf:
            reflected_x = 2 * intercept - x
            return (reflected_x, y)
        reflected_x = (x - 2 * slope * (y - intercept)) / (1 + slope ** 2)
        reflected_y = slope * reflected_x + intercept
        return (reflected_x, reflected_y)

    reflected_pts = [reflect_point(pt) for pt in pts]
    reflected_pts = np.array(reflected_pts).astype(int)

    # Check if the reflected points are close to the original contour points
    distances = [cv2.pointPolygonTest(contour, (x, y), True) for (x, y) in reflected_pts]
    return all(d <= 1 for d in distances)  # Check if all reflected points are close to original contour


# Load image and convert to grayscale
image = cv2.imread(config["image_path"])
if image is None:
    raise FileNotFoundError(f"Image not found at path: {config['image_path']}")
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
    shape_type = 'Contour'  # Default
    if is_circle(smallest_contour):
        draw_circle(shape_image, smallest_contour)
        print_shape_info('Circle', approx, num_edges, area)
        shape_type = 'Circle'
    elif len(approx) == 3:
        draw_polygon(shape_image, approx, color=(0, 255, 0))  # Green for triangles
        print_shape_info('Triangle', approx, num_edges, area)
        shape_type = 'Triangle'
    elif len(approx) == 4:
        rectangle, square = is_rectangle_or_square(approx)
        if square:
            draw_polygon(shape_image, approx, color=(0, 255, 255))  # Cyan for squares
            print_shape_info('Square', approx, num_edges, area)
            shape_type = 'Square'
        elif rectangle:
            draw_polygon(shape_image, approx, color=(255, 0, 0))  # Red for rectangles
            print_shape_info('Rectangle', approx, num_edges, area)
            shape_type = 'Rectangle'
    elif is_star(approx, config["edge_length_tolerance"]):
        draw_polygon(shape_image, approx, color=(0, 0, 255))  # Blue for stars
        print_shape_info('Star', approx, num_edges, area)
        shape_type = 'Star'
    else:
        cv2.drawContours(shape_image, [smallest_contour], 0, (255, 255, 255), 2)  # White for other shapes
        print_shape_info('Contour', approx, num_edges, area)

    # Draw symmetry lines
    draw_symmetry_lines(shape_image, shape_type, approx, smallest_contour)

    # Display the number of edges
    cv2.putText(shape_image, f'Edges: {num_edges}', (approx[0][0][0], approx[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Resize the image for a smaller display window
width = int(image.shape[1] * config["resize_factor"])
height = int(image.shape[0] * config["resize_factor"])
dim = (width, height)
resized_image = cv2.resize(shape_image, dim, interpolation=cv2.INTER_AREA)

# Show the result
cv2.imshow(resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
