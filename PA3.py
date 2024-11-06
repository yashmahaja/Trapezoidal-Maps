class Node:
    def __init__(self, x, y):
        self.x = x  # Can store trapezoid information or line segment reference
        self.y = y
        self.children = []  # Links to child nodes in the DAG

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Trapezoid:
    def __init__(self, top, bottom, left, right):
        self.top = top  # Top bounding segment
        self.bottom = bottom  # Bottom bounding segment
        self.left = left  # Left bounding segment (x-value)
        self.right = right  # Right bounding segment (x-value)
        self.neighbors = {}  # Store neighboring trapezoids if needed


    def __repr__(self):
        return (f"Trapezoid(top={self.top}, bottom={self.bottom}, "
                f"left={self.left}, right={self.right}, "
                f"neighbors={list(self.neighbors.keys())})")


class Solution:
    def __init__(self):
        self.dag = []
        self.trapezoids = []
        self.points = {}
        self.segments = {}
        self.coordinates = {}
        self.min_x = float("inf")
        self.max_x = float("-inf")
        self.min_y = float("inf")
        self.max_y = float("-inf")

    def get_input(self, filename):
        with open(filename, 'r') as f:
            f.readline()
            i = 0
            while True:
                line = f.readline().strip()
                if not line:
                    break
                x1, y1, x2, y2 = map(float, line.split())
                left_label = f"P{i + 1}"
                right_label = f"Q{i + 1}"
                left_label = self.process_point(x1, y1, left_label)
                right_label = self.process_point(x2, y2, right_label)
                self.segments[f"S{i + 1}"] = (left_label, right_label)
                i += 1

        bounding_trapezoid = self.create_bounding_box()
        print("Bounding box:", bounding_trapezoid)
        print("Points:", self.points)
        print("Segments:", self.segments)

    def process_point(self, x, y, label):
        if (x, y) in self.coordinates:
            # Duplicate point, reuse existing label
            existing_label = self.coordinates[(x, y)]
            return existing_label
        else:
            # Unique point, update bounds and store it
            self.update_bounds(x, y)
            new_point = Node(x, y)
            new_point.label = label
            self.points[label] = new_point
            self.coordinates[(x, y)] = label
            return label

    def update_bounds(self, x, y):
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

    def create_bounding_box(self):
        bounding_trapezoid = Trapezoid(
            top=self.max_y + 5,
            bottom=self.min_y - 5,
            left=self.min_x - 5,
            right=self.max_x + 5
        )
        bounding_trapezoid.label = "T1"
        self.trapezoids.append(bounding_trapezoid.label)
        self.dag.append(bounding_trapezoid.left)
        return bounding_trapezoid

    def build_trapezoidal_map(self, segments):
        # Implements the incremental construction algorithm
        pass

    def query_point(self, x, y):
        # Uses the DAG to locate the trapezoidal region containing (x, y)
        pass


def main():
    solution = Solution()

    segments = solution.get_input("InputFiles/ya2390.txt")
    solution.build_trapezoidal_map(segments)



if __name__ == '__main__':
    main()