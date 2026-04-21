# visualize.py
import cv2
import numpy as np

CLASS_NAMES = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure", 5: "Other"}

# Match paper colors exactly (BGR)
CLASS_COLORS = {
    0: (0, 180, 0),      # Text - green
    1: (0, 0, 220),      # Title - red
    2: (0, 140, 255),    # List - orange
    3: (180, 160, 0),    # Table - teal
    4: (180, 0, 180),    # Figure - purple
    5: (120, 120, 120),  # Other - gray
}

# Box thickness per class (Title gets thicker border like paper)
CLASS_THICKNESS = {
    0: 2,   # Text
    1: 3,   # Title - thicker
    2: 2,   # List
    3: 2,   # Table
    4: 2,   # Figure
    5: 1,   # Other
}

def draw_results(image, results, boxes, classes):
    vis = image.copy()

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        color = CLASS_COLORS.get(cls, (180, 180, 180))
        label = CLASS_NAMES.get(cls, str(cls))
        thickness = CLASS_THICKNESS.get(cls, 2)

        # Draw box border only (no fill, no text overlay)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Small label tag at top-left corner of box
        font_scale = 0.38
        font_thickness = 1
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        tag_x2 = x1 + lw + 6
        tag_y2 = y1 + lh + 6

        # Solid color tag background
        cv2.rectangle(vis, (x1, y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(vis, label, (x1 + 3, y1 + lh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Title gets an underline below the box (like paper)
        if cls == 1:
            cv2.line(vis, (x1, y2 + 3), (x2, y2 + 3), color, 2)

    return vis
