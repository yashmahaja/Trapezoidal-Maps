import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class Node:
    def __init__(self, x=None, y=None, type=None, name=None, trapezoid=None, segment=None):
        self.x = x
        self.y = y
        self.type = type
        self.name = name
        self.trapezoid = trapezoid
        self.segment = segment  # For y-nodes
        self.children = []  # Links to child nodes in the DAG

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"Node(type={self.type}, name={self.name})"


class Trapezoid:
    def __init__(self, top_left, bottom_left, top_right, bottom_right,
                 label=None):
        # Corner points of the trapezoid
        self.top_left = top_left
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.bottom_right = bottom_right

        # Equations representing the boundaries of the trapezoid
        self.top_equation = None  # Equation for the top boundary
        self.bottom_equation = None  # Equation for the bottom boundary
        self.left_equation = None  # Equation for the left boundary
        self.right_equation = None  # Equation for the right boundary
        # Dictionary to store neighboring trapezoids if needed
        self.neighbors = {}

        # Label to identify the trapezoid
        self.label = label

    def __repr__(self):
        return (
            f"Trapezoid(label={self.label}, "
            f"top_left={self.top_left}, bottom_left={self.bottom_left}, "
            f"top_right={self.top_right}, bottom_right={self.bottom_right}, "
            f"neighbors={list(self.neighbors.keys())})"
        )


class Solution:
    def __init__(self):
        self.dag = None
        self.trapezoids = {}  # Changed to dictionary
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
            top_left=Node(self.min_x - 5, self.max_y + 5),
            bottom_left=Node(self.min_x - 5, self.min_y - 5),
            top_right=Node(self.max_x + 5, self.max_y + 5),
            bottom_right=Node(self.max_x + 5, self.min_y - 5),
            label="T1"
        )

        self.assign_line_segments(bounding_trapezoid)
        self.trapezoids[bounding_trapezoid.label] = bounding_trapezoid  # Add to dictionary
        self.dag = Node(type='Trapezoid', name='T1', trapezoid=bounding_trapezoid)
        return bounding_trapezoid

    def assign_line_segments(self, trapezoid):
        # Helper function to calculate line equation (A, B, C) from two points
        def line_from_points(p1, p2):
            A = p2.y - p1.y
            B = p1.x - p2.x
            C = A * p1.x + B * p1.y
            return (A, B, C)

        # Calculate and assign line equations based on trapezoid corners
        trapezoid.top_equation = line_from_points(trapezoid.top_left,
                                                  trapezoid.top_right)
        trapezoid.bottom_equation = line_from_points(trapezoid.bottom_left,
                                                     trapezoid.bottom_right)
        trapezoid.left_equation = line_from_points(trapezoid.top_left,
                                                   trapezoid.bottom_left)
        trapezoid.right_equation = line_from_points(trapezoid.top_right,
                                                    trapezoid.bottom_right)

        # Optionally, print the equations for verification
        print(f"Assigned line equations for trapezoid {trapezoid.label}:")
        print(f"  Top equation: {trapezoid.top_equation}")
        print(f"  Bottom equation: {trapezoid.bottom_equation}")
        print(f"  Left equation: {trapezoid.left_equation}")
        print(f"  Right equation: {trapezoid.right_equation}")

    def check_intersections(self, line_segment):
        intersecting_trapezoids = []
        intersection_points = []
        for trapezoid in self.trapezoids.values():  # Adjusted for dictionary
            for boundary_equation in [
                trapezoid.top_equation,
                trapezoid.bottom_equation,
                trapezoid.left_equation,
                trapezoid.right_equation
            ]:
                if self.do_intersect(line_segment, boundary_equation):
                    intersecting_trapezoids.append(trapezoid)
                    intersection_points.append(
                        self.find_intersection_point(line_segment,
                                                     boundary_equation))
        return intersecting_trapezoids, intersection_points

    def do_intersect(self, segment_labels, boundary_equation):
        # segment_labels is in the form (label1, label2), e.g., ("P1", "Q1")
        # boundary_equation is in the form (A, B, C)

        # Unpack the segment labels
        label1, label2 = segment_labels

        # Fetch the actual points for label1 and label2 from the points dictionary
        p1 = self.points[label1]
        q1 = self.points[label2]

        # Unpack the boundary line equation coefficients
        A, B, C = boundary_equation

        # Calculate the "position" of points p1 and q1 relative to the boundary line
        # Substitute p1 and q1 into the line equation Ax + By = C to check on which side they lie
        pos1 = A * p1.x + B * p1.y - C
        pos2 = A * q1.x + B * q1.y - C

        # If pos1 and pos2 have opposite signs, then the segment intersects the line
        if pos1 * pos2 < 0:
            return True

        # Special case: if either pos1 or pos2 is exactly zero, it lies on the boundary line
        if pos1 == 0 or pos2 == 0:
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
        # Retrieve segment endpoints
        left_point = self.points[selected[0]]  # Start point P1
        right_point = self.points[selected[1]]  # End point Q1

        # Ensure left_point.x <= right_point.x
        if left_point.x > right_point.x:
            left_point, right_point = right_point, left_point

        print("Processing segment between", left_point, "and", right_point)

        # Find trapezoids containing the segment endpoints
        result1, parent1 = self.where_is_my_point((left_point.x, left_point.y))
        result2, parent2 = self.where_is_my_point((right_point.x, right_point.y))

        # Since both points lie in the same trapezoid
        if result1 == result2:
            print("Both points lie within the same trapezoid.")

            old_trapezoid = result1.trapezoid  # Get the actual trapezoid object

            x1 = left_point.x
            x2 = right_point.x

            # Compute intersection points with the top boundary
            if old_trapezoid.top_right.x != old_trapezoid.top_left.x:
                m_top = (old_trapezoid.top_right.y - old_trapezoid.top_left.y) / (
                    old_trapezoid.top_right.x - old_trapezoid.top_left.x)
                y1_top = old_trapezoid.top_left.y + m_top * (x1 - old_trapezoid.top_left.x)
                y2_top = old_trapezoid.top_left.y + m_top * (x2 - old_trapezoid.top_left.x)
            else:
                y1_top = old_trapezoid.top_left.y
                y2_top = old_trapezoid.top_left.y

            # Compute intersection points with the bottom boundary
            if old_trapezoid.bottom_right.x != old_trapezoid.bottom_left.x:
                m_bottom = (old_trapezoid.bottom_right.y - old_trapezoid.bottom_left.y) / (
                    old_trapezoid.bottom_right.x - old_trapezoid.bottom_left.x)
                y1_bottom = old_trapezoid.bottom_left.y + m_bottom * (x1 - old_trapezoid.bottom_left.x)
                y2_bottom = old_trapezoid.bottom_left.y + m_bottom * (x2 - old_trapezoid.bottom_left.x)
            else:
                y1_bottom = old_trapezoid.bottom_left.y
                y2_bottom = old_trapezoid.bottom_left.y

            # Create the four new trapezoids
            # Trapezoid T1 (Left of the segment)
            T1 = Trapezoid(
                top_left=old_trapezoid.top_left,
                top_right=Node(x1, y1_top),
                bottom_left=old_trapezoid.bottom_left,
                bottom_right=Node(x1, y1_bottom),
                label='T1'
            )
            self.assign_line_segments(T1)

            # Trapezoid T4 (Right of the segment)
            T4 = Trapezoid(
                top_left=Node(x2, y2_top),
                top_right=old_trapezoid.top_right,
                bottom_left=Node(x2, y2_bottom),
                bottom_right=old_trapezoid.bottom_right,
                label='T4'
            )
            self.assign_line_segments(T4)

            # Trapezoid T2 (Above the segment between the vertical lines)
            T2 = Trapezoid(
                top_left=T1.top_right,
                top_right=T4.top_left,
                bottom_left=left_point,
                bottom_right=right_point,
                label='T2'
            )
            self.assign_line_segments(T2)

            # Trapezoid T3 (Below the segment between the vertical lines)
            T3 = Trapezoid(
                top_left=left_point,
                top_right=right_point,
                bottom_left=T1.bottom_right,
                bottom_right=T4.bottom_left,
                label='T3'
            )
            self.assign_line_segments(T3)

            # Update the DAG
            # Create nodes for the trapezoids
            T1_node = Node(type='Trapezoid', name='T1', trapezoid=T1)
            T2_node = Node(type='Trapezoid', name='T2', trapezoid=T2)
            T3_node = Node(type='Trapezoid', name='T3', trapezoid=T3)
            T4_node = Node(type='Trapezoid', name='T4', trapezoid=T4)

            # Create nodes for x and y tests
            x_node_left = Node(x=x1, type='xnode', name=f'x={x1}')
            x_node_right = Node(x=x2, type='xnode', name=f'x={x2}')
            y_node = Node(type='ynode', name='Segment', segment=(left_point, right_point))

            # Build the new DAG structure
            x_node_left.children = [T1_node, x_node_right]
            x_node_right.children = [y_node, T4_node]
            y_node.children = [T2_node, T3_node]  # [Above, Below]

            # Replace the old trapezoid node in the DAG
            if parent1:
                for i, child in enumerate(parent1.children):
                    if child == result1:
                        parent1.children[i] = x_node_left
                        break
            else:
                # If the old trapezoid was the root
                self.dag = x_node_left

            # Update the trapezoids dictionary
            del self.trapezoids[old_trapezoid.label]  # Remove old trapezoid
            self.trapezoids[T1.label] = T1
            self.trapezoids[T2.label] = T2
            self.trapezoids[T3.label] = T3
            self.trapezoids[T4.label] = T4

            print("Segment insertion complete.")

        else:
            print("Points do not lie within the same trapezoid.")

    def get_trapezoids(self):
        pass

    def x_test(self, node, x_value):
        # return True if x_value > node.x (go to right child), else False
        return x_value > node.x

    def y_test(self, node, point):
        left_point = node.segment[0]
        right_point = node.segment[1]

        if right_point.x != left_point.x:
            slope = (right_point.y - left_point.y) / (right_point.x - left_point.x)
            intercept = left_point.y - slope * left_point.x
        else:
            return point[1] > left_point.y  # For vertical lines

        y_on_line = slope * point[0] + intercept

        return point[1] > y_on_line  # True if point is above the segment

    def where_is_my_point(self, point):

        # Traverse DAG.
        node = self.dag
        parent = None
        while len(node.children) != 0:
            if node.type == 'xnode':
                result = self.x_test(node, point[0])
                parent = node
                if result:
                    node = node.children[1]
                else:
                    node = node.children[0]
            elif node.type == 'ynode':
                result = self.y_test(node, point)
                parent = node
                if result:
                    node = node.children[0]  # Above segment
                else:
                    node = node.children[1]  # Below segment
            else:
                # Trapezoid node
                break

        return node, parent

    def build_trapezoidal_map(self):

        # Randomly add a segment
        selected = self.segments['S1']
        print("Selected:", selected)

        self.insert_segments(selected)
        # Implements the incremental construction algorithm

    def query_point(self, x, y):
        # Uses the DAG to locate the trapezoidal region containing (x, y)
        pass
class TrapezoidalMapVisualizer:
    def __init__(self, trapezoids, segments, points, min_x, max_x, min_y, max_y):
        self.trapezoids = trapezoids
        self.segments = segments
        self.points = points
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def plot(self):
        fig, ax = plt.subplots()
        # Plot trapezoids
        for label, trapezoid in self.trapezoids.items():
            # Get corner points
            points = [
                (trapezoid.top_left.x, trapezoid.top_left.y),
                (trapezoid.top_right.x, trapezoid.top_right.y),
                (trapezoid.bottom_right.x, trapezoid.bottom_right.y),
                (trapezoid.bottom_left.x, trapezoid.bottom_left.y)
            ]
            # Create a polygon patch
            polygon = Polygon(points, closed=True, fill=None, edgecolor='blue', linewidth=1)
            ax.add_patch(polygon)
            # Annotate the trapezoid label at its centroid
            centroid_x = sum([p[0] for p in points]) / 4
            centroid_y = sum([p[1] for p in points]) / 4
            ax.text(centroid_x, centroid_y, label, color='blue', fontsize=8)

            # Print trapezoid coordinates
            print(f"Trapezoid {label} coordinates:")
            for i, point in enumerate(points):
                print(f"  Point {i+1}: ({point[0]}, {point[1]})")
            print()

        # Plot segments
        for label, (start_label, end_label) in self.segments.items():
            start_point = self.points[start_label]
            end_point = self.points[end_label]
            x_values = [start_point.x, end_point.x]
            y_values = [start_point.y, end_point.y]
            ax.plot(x_values, y_values, color='red', linewidth=2, label=f'Segment {label}')
            # Print segment coordinates
            print(f"Segment {label} from ({start_point.x}, {start_point.y}) to ({end_point.x}, {end_point.y})")

        # Set plot limits
        ax.set_xlim(self.min_x - 10, self.max_x + 10)
        ax.set_ylim(self.min_y - 10, self.max_y + 10)

        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Trapezoidal Map')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.show()

def get_plotting_data(solution, inserted_segments):
    # Filter segments to include only those in inserted_segments
    segments = {k: v for k, v in solution.segments.items() if k in inserted_segments}
    plotting_data = {
        'trapezoids': solution.trapezoids,
        'segments': segments,
        'points': solution.points,
        'min_x': solution.min_x,
        'max_x': solution.max_x,
        'min_y': solution.min_y,
        'max_y': solution.max_y
    }
    return plotting_data

def main():
    solution = Solution()
    solution.get_input("InputFiles/ya2390.txt")
    solution.build_trapezoidal_map()

    # List of segments that have been inserted into the trapezoidal map
    inserted_segments = ['S1']  # Update this list as you insert more segments

    # Get plotting data with only inserted segments
    plotting_data = get_plotting_data(solution, inserted_segments)

    # Initialize visualizer with plotting data
    visualizer = TrapezoidalMapVisualizer(
        trapezoids=plotting_data['trapezoids'],
        segments=plotting_data['segments'],
        points=plotting_data['points'],
        min_x=plotting_data['min_x'],
        max_x=plotting_data['max_x'],
        min_y=plotting_data['min_y'],
        max_y=plotting_data['max_y']
    )

    # Plot the trapezoidal map
    visualizer.plot()


if __name__ == '__main__':
    main()
