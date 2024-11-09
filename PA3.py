class Node:
    def __init__(self, x= None, y= None, type = None,name = None):
        self.x = x  # Can store trapezoid information or line segment reference
        self.y = y
        self.type = type
        self.name = name
        self.children = []  # Links to child nodes in the DAG

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Trapezoid:
    def __init__(self, top_left, bottom_left, top_right, bottom_right,
                 label=None):
        # Corner points of the trapezoid
        self.top_left = top_left
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.bottom_right = bottom_right

        # Segments representing the boundaries of the trapezoid
        self.top_segment = None  # Top boundary segment (e.g., horizontal line)
        self.bottom_segment = None  # Bottom boundary segment
        self.left_segment = None  # Left vertical boundary
        self.right_segment = None  # Right vertical boundary

        # Dictionary to store neighboring trapezoids if needed
        self.neighbors = {}

        # Label to identify the trapezoid
        self.label = label

    def __repr__(self):
        return (
            f"Trapezoid(top_left={self.top_left}, bottom_left={self.bottom_left}, "
            f"top_right={self.top_right}, bottom_right={self.bottom_right}, "
            f"neighbors={list(self.neighbors.keys())}, label={self.label})")


class Solution:
    def __init__(self):
        self.dag = None
        self.trapezoids = []
        self.points = {}
        self.segments = {}
        self.coordinates = {}
        self.points_to_label = {}
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
            # new_point.label = label
            self.points[label] = new_point
            self.points_to_label[new_point] = label
            self.coordinates[(x, y)] = label
            return label

    def update_bounds(self, x, y):
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

    def create_bounding_box(self):
        bounding_trapezoid = Trapezoid(
            top_left = Node(self.min_x - 5 , self.max_y +5),
            bottom_left = Node(self.min_x -5, self.min_y - 5),
            top_right = Node(self.max_x + 5, self.max_y +5 ),
            bottom_right = Node(self.max_x +5, self.min_y -5)
        )

        bounding_trapezoid.label = "T1"
        self.trapezoids.append(bounding_trapezoid.label)
        self.dag = Node( type='Trapezoid', name='T1')
        # self.dag.append(bounding_trapezoid.left)
        return bounding_trapezoid

    def check_intersections(self, line_segment):
        intersecting_trapezoids = []
        intersection_points = []
        for trapezoid in self.trapezoids:
            for boundary in [trapezoid.top_segment, trapezoid.bottom_segment,
                             trapezoid.left_segment, trapezoid.right_segment]:
                if self.do_intersect(line_segment, boundary):
                    intersecting_trapezoids.append(trapezoid)
                    intersection_points.append(
                        self.find_intersection_point(line_segment, boundary))
        return intersecting_trapezoids, intersection_points

    def do_intersect(self, segment1, segment2):
        # Helper function to check if two line segments intersect
        def orientation(p, q, r):
            val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
            if val == 0:
                return 0  # collinear
            return 1 if val > 0 else 2  # clock or counterclock wise

        p1, q1 = segment1
        p2, q2 = segment2

        # Find the four orientations needed for the general and special cases
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True

        # p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True

        # p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True

        # p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True

        return False

    def on_segment(self, p, q, r):
        if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
            return True
        return False

    def find_intersection_point(self, segment1, segment2):
        # Helper function to find the intersection point of two line segments
        p1, q1 = segment1
        p2, q2 = segment2

        A1 = q1.y - p1.y
        B1 = p1.x - q1.x
        C1 = A1 * p1.x + B1 * p1.y

        A2 = q2.y - p2.y
        B2 = p2.x - q2.x
        C2 = A2 * p2.x + B2 * p2.y

        determinant = A1 * B2 - A2 * B1

        if determinant == 0:
            return None  # The lines are parallel (should not happen in general case)

        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant

        return Node(x, y)

    def insert_segments(self, selected):



        left_point  = self.points[selected[0]]
        right_point = self.points[selected[1]]

        # if (self.dag == None):
        #     self.dag = Node(left_point[0], left_point[1], type = 'xnode')

        print(left_point , right_point)
        result1, parent1 = self.where_is_my_point(left_point)
        result2, parent2 = self.where_is_my_point(right_point)

        self.check_intersections(selected)

        if result1 == result2: # Same trapezoid.
            # Break into 4 - replace the dag node with
            parent1.children[]



        pass

    def where_is_my_point(self, left_point):

        #Traverse Dag.

        # At every node - if its a x node- > call xtest
        # If y node call y test
        # Based on the return value you traverse further.

        node = self.dag
        parent = None
        while len(node.children != 0):
            if node.type== 'xnode':
                result = self.xtest(node, left_point[0])
                parent = node
                if result:
                    node = node.children[1]
                else:
                    node = node.children[0]
            else:
                self.y_test(node, left_point)

        return node,parent

    def x_test(self, node, x_value ):
        # return true if right , false if left.
        return x_value > node.x
    def y_test(self, node, left_point):

        # node_label = points_to_label[]

        # self.segments[node]
        return True


    def build_trapezoidal_map(self):

        # Randomly Add segment
        selected = self.segments['S1']
        print("Selected:" , selected)


        self.insert_segments(selected)
        # Implements the incremental construction algorithm
        pass

    def query_point(self, x, y):
        # Uses the DAG to locate the trapezoidal region containing (x, y)
        pass


def main():
    solution = Solution()

    solution.get_input("InputFiles/ya2390.txt")
    solution.build_trapezoidal_map()



if __name__ == '__main__':
    main()