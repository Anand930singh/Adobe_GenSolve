import numpy as np
import svgwrite
import cairosvg
from io import BytesIO

def read_csv(csv_path):
    # Read CSV data
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')

    path_XYs = []

    # Process data
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)

    return path_XYs

def polylines2svg(paths_XYs, svg_path, colours):
    # Calculate the bounding box dimensions with padding
    svg_path = svg_path.replace('problems', 'png_svg')
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W = max(W, np.max(XY[:, 0]))
            H = max(H, np.max(XY[:, 1]))

    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    # Create a new SVG drawing with the size
    svg_path = svg_path.replace('.csv', '.svg')
    dwg = svgwrite.Drawing(svg_path, size=(W, H), profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    for i, path_XYs in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]

        for XY in path_XYs:
            path_data.append("M{},{}".format(XY[0, 0], XY[0, 1]))
            for j in range(1, len(XY)):
                path_data.append("L{},{}".format(XY[j, 0], XY[j, 1]))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append("Z")

        path_str = " ".join(path_data)
        group.add(dwg.path(d=path_str, fill='white', stroke=c, stroke_width=2))

    dwg.add(group)
    dwg.save()

    # Convert SVG to PNG in memory
    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(
        url=svg_path,
        write_to=png_path,
        parent_width=W,
        parent_height=H,
        output_width=fact * W,
        output_height=fact * H,
        background_color='white'
    )

    return png_path

colours = ['black']
image_csv_path = './problems/frag2.csv'
image = polylines2svg(read_csv(image_csv_path), image_csv_path, colours)
