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
        # self.dag = DAG()  # DAG to represent the trapezoidal map
        self.trapezoids = []  # Store trapezoids for the decomposition

    def get_input(self, filename):
        segments = []
        with open(filename, 'r') as f:
            n = int(f.readline().strip())
            for _ in range(n):
                x1, y1, x2, y2 = map(float, f.readline().strip().split())
                segment = (Node(x1,y1), Node(x2,y2))
                segments.append(segment)
        print(segments)
        self.bounding_box(segments)
        return segments

    def bounding_box(self,segments):
        min_x = min(segments[0][0].x, segments[0][1].x)
        max_x = max(segments[0][0].x, segments[0][1].x)
        min_y = min(segments[0][0].y, segments[0][1].y)
        max_y = max(segments[0][0].y, segments[0][1].y)

        for segment in segments:
            for node in segment:
                min_x = min(min_x, node.x)
                max_x = max(max_x, node.x)
                min_y = min(min_y, node.y)
                max_y = max(max_y, node.y)
        bounding_trapezoid = Trapezoid(top=max_y + 5, bottom=min_y - 5, left=min_x - 5 , right=max_x + 5)
        print("Bounding box:", bounding_trapezoid)
        return bounding_trapezoid

    def build_trapezoidal_map(self, segments):
        # Implements the incremental construction algorithm
        pass

    def query_point(self, x, y):
        # Uses the DAG to locate the trapezoidal region containing (x, y)
        pass


def main():
    solution = Solution()

    segments = solution.get_input("InputFiles/ya2390.txt")  # Replace with the actual input file name
    solution.build_trapezoidal_map(segments)


if __name__ == '__main__':
    main()