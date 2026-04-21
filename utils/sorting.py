# sorting.py
def sort_boxes(boxes, classes):
    data = list(zip(boxes, classes))
    data.sort(key=lambda x: (x[0][1], x[0][0]))
    return data