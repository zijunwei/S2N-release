class Interval(object):
    def __init__(self, start_position, end_position, confidence):

        self.size = end_position - start_position + 1
        self.start_position = start_position
        self.end_position = end_position
        self.confidence = confidence

    def overlaps(self, other):
        pass

    def distance(self, other):
        pass
        # return sum(map(abs, [self.cx - other.cx, self.cy - other.cy,
        #                self.width - other.width, self.height - other.height]))

    def intersection(self, other):
        pass

        # left = max(self.cx - self.width/2., other.cx - other.width/2.)
        # right = min(self.cx + self.width/2., other.cx + other.width/2.)
        # width = max(right - left, 0)
        # top = max(self.cy - self.height/2., other.cy - other.height/2.)
        # bottom = min(self.cy + self.height/2., other.cy + other.height/2.)
        # height = max(bottom - top, 0)
        # return width * height
    def size(self):
        return self.size

    def union(self, other):
        pass

        # return self.area() + other.area() - self.intersection(other)


    def iou(self, other):
        pass

        # return self.intersection(other) / self.union(other)
    def __eq__(self, other):
        return (self.start_position == other.start_position and
            self.end_position == other.end_position and
            self.confidence == other.confidence)