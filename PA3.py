"""
Authors: Yash Mahajan, Yash Awaghate
This script implements a trapezoidal map for point location queries using a randomized incremental construction algorithm.
"""
import matplotlib.pyplot as plt
import pandas as pd
import collections


root_node = None

class Trapezoid:
    _counter = 0  # Class-level attribute for tracking the next available ID

    def __init__(self, top, bottom, left_point, right_point):
        self.top = top
        self.bottom = bottom
        self.leftp = left_point
        self.rightp = right_point
        self.trapezoid_id = self._generate_id()
        self.identifier = f"T{self.trapezoid_id}"  # Assigning a unique identifier

        self.neighbors = self._initialize_neighbors()

    @classmethod
    def _generate_id(cls):
        """
        Generates a unique ID for each trapezoid instance.
        """
        current_id = cls._counter
        cls._counter += 1  # Incrementing the counter for the next trapezoid
        return current_id

    def _initialize_neighbors(self):
        """
        Initializes the neighbors dictionary with default None values.
        """
        return {
            'top_left': None,
            'bottom_left': None,
            'top_right': None,
            'bottom_right': None
        }

    def set_neighbor(self, position, neighbor):
        """
        Sets a neighbor for the trapezoid at the specified position.

        :param position: One of 'top_left', 'bottom_left', 'top_right', 'bottom_right'.
        :param neighbor: The neighboring trapezoid to set.
        """
        if position in self.neighbors:
            self.neighbors[position] = neighbor
        else:
            raise ValueError(f"Invalid neighbor position: {position}")

    def get_neighbor(self, position):
        """
        Retrieves the neighbor at the specified position.

        :param position: One of 'top_left', 'bottom_left', 'top_right', 'bottom_right'.
        :return: The neighboring trapezoid at the specified position.
        """
        return self.neighbors.get(position)

    def __repr__(self):
        return (f"Trapezoid(top={self.top}, bottom={self.bottom}, leftp={self.leftp}, "
                f"rightp={self.rightp}, trapezoid_id={self.trapezoid_id}, "
                f"identifier='{self.identifier}', neighbors={self.neighbors})")

# these dictionaries are displayed at the end for debugging and to get a clear idea
# of the data structures to manage the relationships between nodes

class Point:
    def __init__(self, x, y, is_left=True):
        self.x = x
        self.y = y
        self.identifier = self._assign_identifier(is_left)


    _left_counter = 0
    _right_counter = 0
    @classmethod
    def _assign_identifier(cls, is_left):
        identifier = f"{'P' if is_left else 'Q'}{cls._left_counter if is_left else cls._right_counter}";
        cls._left_counter += is_left; cls._right_counter += not is_left; return identifier


    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, id='{self.identifier}')"

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class Segment:
    _counter = 1

    def __init__(self, left_point, right_point):
        self.left = left_point
        self.right = right_point
        self.identifier = self._assign_identifier()
        self._validate_points()

    @classmethod
    def _assign_identifier(cls):
        identifier = f"S{cls._counter}"
        cls._counter += 1
        return identifier

    def _validate_points(self):
        if self.left.x > self.right.x:
            raise ValueError("Left point must have a smaller x-coordinate than the right point.")

    def __repr__(self):
        return f"Segment(left={self.left}, right={self.right}, id='{self.identifier}')"

    def get_slope(self):
        dx = self.right.x - self.left.x
        if dx == 0:
            return float('inf')
        return (self.right.y - self.left.y) / dx

    def get_y_at(self, x):
        slope = self.get_slope()
        return self.left.y + slope * (x - self.left.x)


def display_dictionaries():
    dictionaries = {'A (Left Node)': A, 'B (Right Node)': B, 'C (Segment)': C, 'D (Trapezoid)': D}
    for name, dictionary in dictionaries.items():
        print(f"\n{name}:")
        for key, value in dictionary.items():
            print(f"  {key}: {value}")



class Node:
    def __init__(self):
        self.parents = []

    def add_parent(self, parent):
        self.parents.append(parent)

    def remove_parents(self):
        self.parents = []


class XNode(Node):
    def __init__(self, x_value, point_represented, left=None, right=None):
        super().__init__()
        self.x_value = x_value
        self.point_represented = point_represented
        self.left = left
        self.right = right

    def __repr__(self):
        return f"XNode(x_value={self.x_value}, point={self.point_represented})"


class YNode(Node):
    def __init__(self, segment, up=None, down=None):
        super().__init__()
        self.segment = segment
        self.above = up
        self.below = down

    def __repr__(self):
        return f"YNode(segment={self.segment})"


class LeafNode(Node):
    def __init__(self, trapezoid_id):
        super().__init__()
        self.trapezoid_id = trapezoid_id

    def reconnect_parents_to_new_subtree(self, new_subtree_root):
        for parent in self.parents:
            if isinstance(parent, XNode):
                setattr(parent, 'left' if parent.left == self else 'right', new_subtree_root)
            elif isinstance(parent, YNode):
                setattr(parent, 'above' if parent.above == self else 'below', new_subtree_root)
            new_subtree_root.add_parent(parent)
        self.remove_parents()  # Clear parent references


    def __repr__(self):
        return f"LeafNode(trapezoid_id={self.trapezoid_id})"

# Four global dictionaries to aid in constructing the adjacency matrix:
# Dictionary `A` maps left node identifiers to XNode instances, each associated with a unique Point object.
A = B = C = D = collections.defaultdict(list)

# Dictionary `B` maps right node identifiers to XNode instances, each linked to a single Point object.
B = collections.defaultdict(list)

# Dictionary `C` associates segment identifiers with YNode instances, each tied to a unique Segment object.
C = collections.defaultdict(list)

# Dictionary `D` maps trapezoid identifiers to LeafNode instances, each referencing a Trapezoid object through the mapping.
D = collections.defaultdict(list)

def process(filename):
    segments, bounding_box = read_input_file(filename)
    return segments, bounding_box

def read_input_file(filename):
    segments, bounding_box = [], []
    try:
        with open(filename, 'r') as file:
            num_segments = int(file.readline().strip())
            bounding_box = [int(x) for x in file.readline().split()]
            segments = [[int(x) for x in file.readline().split()] for _ in range(num_segments)]
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except ValueError as e:
        print(f"Error while processing the file '{filename}': {e}")
    return segments, bounding_box




def add_totals(df):
    df.loc['Total'] = df.iloc[:-1].sum()
    df['Total'] = df.iloc[:, :-1].sum(axis=1)
    return df

def show_matrices():
    data = {
        'Column1': [1, 2, 3],
        'Column2': [4, 5, 6],
        'Column3': [7, 8, 9]
    }

    df = pd.DataFrame(data, columns=['Col1', 'Col2', 'Col3'], index=['R1', 'R2', 'R3'])
    df = add_totals(df)
    df.at['Row1', 'Column2'] = 500

    df = add_totals(df)

    print(df)






def is_point_above_segment(point, segment):
    if segment.left.x == segment.right.x:
        return point.y > max(segment.left.y, segment.right.y)
    return point.y > (segment.left.y + ((segment.right.y - segment.left.y) / (segment.right.x - segment.left.x)) * (point.x - segment.left.x))



def query_point_lies_on_segment(point, segment):
    x1, y1, x2, y2 = segment.left.x, segment.left.y, segment.right.x, segment.right.y
    return (x2 - x1) * (point.y - y1) == (y2 - y1) * (point.x - x1) and min(x1, x2) <= point.x <= max(x1, x2) and min(y1, y2) <= point.y <= max(y1, y2)


def compare_segment_slopes(segment1, segment2, point):
    slope1 = (segment1.right.y - segment1.left.y) / (
                segment1.right.x - segment1.left.x) if segment1.right.x != segment1.left.x else float('inf')
    slope2 = (segment2.right.y - segment2.left.y) / (
                segment2.right.x - segment2.left.x) if segment2.right.x != segment2.left.x else float('inf')

    return slope1 > slope2


def query(start_node, point, active_segment=None):
    node = start_node
    while not isinstance(node, LeafNode):
        if isinstance(node, XNode):
            node = node.right if point.x <= node.x_value else node.left
        elif isinstance(node, YNode):
            seg = node.segment
            if query_point_lies_on_segment(point, seg):
                node = node.above if compare_segment_slopes(active_segment, seg, point) else node.below
            else:
                node = node.above if is_point_above_segment(point, seg) else node.below

    return node



def find_intersected_trapezoids(segment, trapezoids):
    global root_node
    intersected_trapezoids = []

    # Step 1: Identify endpoints
    left_endpoint, right_endpoint = segment.left, segment.right

    display_dictionaries()

    # Step 2: Find the first trapezoid that contains the left endpoint
    current_trapezoid_node = query(root_node, left_endpoint, segment)
    print("Current leaf node is: ", current_trapezoid_node)

    current_trapezoid = get_trapezoid_from_node(current_trapezoid_node, trapezoids)

    # Step 3: Iteratively find neighboring trapezoids
    while current_trapezoid and right_endpoint.x > current_trapezoid.rightp.x:
        intersected_trapezoids.append(current_trapezoid.trapezoid_id)
        current_trapezoid = move_to_next_trapezoid(current_trapezoid, segment)

    # Add the last trapezoid if it exists
    if current_trapezoid:
        intersected_trapezoids.append(current_trapezoid.trapezoid_id)

    return intersected_trapezoids

def get_trapezoid_from_node(trapezoid_node, trapezoids):
    trapezoid_id = trapezoid_node.trapezoid_id
    # Assuming the third element is the Trapezoid object
    trapezoid = trapezoids[trapezoid_id][2]
    return trapezoid

def move_to_next_trapezoid(current_trapezoid, segment):
    if is_point_above_segment(current_trapezoid.rightp, segment):
        # Move to lower right neighbor
        next_trapezoid = current_trapezoid.neighbors['bottom_right']
    else:
        # Move to upper right neighbor
        next_trapezoid = current_trapezoid.neighbors['top_right']
    return next_trapezoid



def update_neighbors(intersected_trapezoid, new_trapezoid):
    """
    Updates the neighbors of the new trapezoid based on the neighbors of the intersected trapezoid.
    """
    # Iterate over each neighbor of the intersected trapezoid
    for side, neighbor in intersected_trapezoid.neighbors.items():
        if neighbor is not None:
            if side in ["top_left", "bottom_left"]:
                # For left side neighbors
                update_neighbor_side(new_trapezoid, neighbor, side, left_side=True)
            else:
                # For right side neighbors
                update_neighbor_side(new_trapezoid, neighbor, side, left_side=False)


def update_neighbor_side(new_trapezoid, neighbor, side, left_side):
    """
    Helper function to update the neighbor on a specific side (left or right).
    """
    if left_side:
        top_key = "top_left"
        bottom_key = "bottom_left"
    else:
        top_key = "top_right"
        bottom_key = "bottom_right"

    # Update top neighbor if applicable
    if new_trapezoid.top == neighbor.top and new_trapezoid.neighbors.get(top_key) is None:
        new_trapezoid.neighbors[top_key] = neighbor

    # Update bottom neighbor if applicable
    if new_trapezoid.bottom == neighbor.bottom and new_trapezoid.neighbors.get(bottom_key) is None:
        new_trapezoid.neighbors[bottom_key] = neighbor


# ****************************************************************************

def update(trapezoids_intersected, trapezoids, left_point, right_point, segment):
    global root_node

    for trap_id in trapezoids_intersected:
        if trap_id not in trapezoids:
            continue

        trap_identifier, trap_leaf, trapezoid = trapezoids[trap_id]

        if left_point.x < trapezoid.leftp.x and trapezoid.rightp.x < right_point.x:
            handle_case_segment_crosses_trapezoid(trapezoids, trapezoid, trap_leaf, segment)
        elif left_point.x >= trapezoid.leftp.x and right_point.x <= trapezoid.rightp.x:
            handle_case_segment_lies_inside_trapezoid(trapezoids, trapezoid, trap_leaf, segment, left_point,
                                                      right_point)
        else:
            handle_case_one_point_inside_one_point_outside(trapezoids, trapezoid, trap_leaf, segment, left_point,
                                                           right_point)


def handle_case_segment_crosses_trapezoid(trapezoids, intersected_trapezoid, intersected_trapezoid_leaf, segment):
    global D, C

    # Create and link two new trapezoids
    top_trap = create_trapezoid(intersected_trapezoid.top, None, intersected_trapezoid.leftp, intersected_trapezoid.rightp)
    bottom_trap = create_trapezoid(None, intersected_trapezoid.bottom, intersected_trapezoid.leftp, intersected_trapezoid.rightp)
    top_trap.bottom, bottom_trap.top = bottom_trap, top_trap

    # Update neighbors and remove old trapezoid
    update_neighbors(intersected_trapezoid, top_trap)
    update_neighbors(intersected_trapezoid, bottom_trap)
    remove_trapezoid(trapezoids, D, intersected_trapezoid)

    # Create leaf nodes and a Y-node
    top_leaf, bottom_leaf = create_leaf_node(top_trap), create_leaf_node(bottom_trap)
    segment_node = YNode(segment, top_leaf, bottom_leaf)
    C[segment.identifier].append(segment_node)

    # Update parent pointers
    top_leaf.parents.append(segment_node)
    bottom_leaf.parents.append(segment_node)

    # Add new trapezoids and update search structure
    add_trapezoid(trapezoids, top_trap, top_leaf)
    add_trapezoid(trapezoids, bottom_trap, bottom_leaf)
    intersected_trapezoid_leaf.reconnect_parents_to_new_subtree(segment_node)


def handle_case_segment_lies_inside_trapezoid(trapezoids, intersected_trapezoid, intersected_trapezoid_leaf, segment, left_point, right_point):
    global D, C, A, B, root_node

    # Create and link new trapezoids
    U_trap = create_trapezoid(intersected_trapezoid.top, intersected_trapezoid.bottom, intersected_trapezoid.leftp, left_point)
    Y_trap = create_trapezoid(intersected_trapezoid.top, None, left_point, right_point)
    Z_trap = create_trapezoid(None, intersected_trapezoid.bottom, left_point, right_point)
    X_trap = create_trapezoid(intersected_trapezoid.top, intersected_trapezoid.bottom, right_point, intersected_trapezoid.rightp)

    # Link relationships
    Y_trap.bottom, Z_trap.top = Z_trap, Y_trap

    # Update neighbors and remove the old trapezoid
    [update_neighbors(intersected_trapezoid, trap) for trap in [U_trap, X_trap]]
    set_neighbors_case_segment_lies_inside(U_trap, Y_trap, Z_trap, X_trap)
    remove_trapezoid(trapezoids, D, intersected_trapezoid)

    # Create leaf nodes and connections
    U_leaf, Y_leaf, Z_leaf, X_leaf = (create_leaf_node(trap) for trap in [U_trap, Y_trap, Z_trap, X_trap])
    y_node = YNode(segment, Y_leaf, Z_leaf)
    C[segment.identifier].append(y_node)

    p_node = XNode(left_point.x, left_point, U_leaf, XNode(right_point.x, right_point, y_node, X_leaf))
    A[left_point.identifier].append(p_node)
    B[right_point.identifier].append(p_node.right)

    # Set parent references
    for leaf, parent in [(U_leaf, p_node), (Y_leaf, y_node), (Z_leaf, y_node), (X_leaf, p_node.right)]:
        leaf.parents.append(parent)

    # Add new trapezoids
    [add_trapezoid(trapezoids, trap, leaf) for trap, leaf in [(U_trap, U_leaf), (Y_trap, Y_leaf), (Z_trap, Z_leaf), (X_trap, X_leaf)]]

    # Reconnect and update root if necessary
    intersected_trapezoid_leaf.reconnect_parents_to_new_subtree(p_node)
    if isinstance(root_node, LeafNode):
        root_node = p_node



def handle_case_one_point_inside_one_point_outside(trapezoids, intersected_trapezoid, intersected_trapezoid_leaf,
                                                   segment, left_point, right_point):
    global D, C, A, B

    # Create and link new trapezoids for splitting
    traps = [
        create_trapezoid(intersected_trapezoid.top, intersected_trapezoid.bottom, intersected_trapezoid.leftp,
                         left_point),
        create_trapezoid(intersected_trapezoid.top, None, left_point, right_point),
        create_trapezoid(None, intersected_trapezoid.bottom, left_point, intersected_trapezoid.rightp)
    ]
    traps[1].bottom, traps[2].top = traps[2], traps[1]  # Link top and bottom

    set_neighbors_case_one_point_inside(*traps)
    [update_neighbors(intersected_trapezoid, trap) for trap in traps]
    remove_trapezoid(trapezoids, D, intersected_trapezoid)

    # Create nodes and update structure
    leaves = [create_leaf_node(trap) for trap in traps]
    segment_y_node = YNode(segment, leaves[1], leaves[2])
    C[segment.identifier].append(segment_y_node)

    subtree_root = XNode(left_point.x, left_point, leaves[0],
                         segment_y_node) if left_point.x >= intersected_trapezoid.leftp.x else XNode(right_point.x,
                                                                                                     right_point,
                                                                                                     leaves[0],
                                                                                                     segment_y_node)
    (A if left_point.x >= intersected_trapezoid.leftp.x else B)[
        left_point.identifier if left_point.x >= intersected_trapezoid.leftp.x else right_point.identifier].append(
        subtree_root)
    leaves[0].parents.append(subtree_root)

    for leaf in leaves[1:]:
        leaf.parents.append(segment_y_node)
    [add_trapezoid(trapezoids, trap, leaf) for trap, leaf in zip(traps, leaves)]

    intersected_trapezoid_leaf.reconnect_parents_to_new_subtree(subtree_root)


def create_trapezoid(top, bottom, leftp, rightp):
    trapezoid = Trapezoid(top, bottom, leftp, rightp)
    return trapezoid

def create_leaf_node(trapezoid):
    leaf = LeafNode(trapezoid.trapezoid_id)
    D[trapezoid.identifier].append(leaf)
    return leaf

def remove_trapezoid(trapezoids, T, trapezoid):
    trapezoids.pop(trapezoid.trapezoid_id)
    T.pop(trapezoid.identifier)

def add_trapezoid(trapezoids, trapezoid, leaf):
    trapezoids[trapezoid.trapezoid_id] = [trapezoid.identifier, leaf, trapezoid]

def set_neighbors_case_segment_lies_inside(U_trapezoid, Y_trapezoid, Z_trapezoid, X_trapezoid):
    # Set neighbors for U, Y, Z, X trapezoids
    U_trapezoid.neighbors["top_right"] = Y_trapezoid
    U_trapezoid.neighbors["bottom_right"] = Z_trapezoid

    Y_trapezoid.neighbors["top_left"] = U_trapezoid
    Y_trapezoid.neighbors["top_right"] = X_trapezoid

    Z_trapezoid.neighbors["bottom_left"] = U_trapezoid
    Z_trapezoid.neighbors["bottom_right"] = X_trapezoid

    X_trapezoid.neighbors["top_left"] = Y_trapezoid
    X_trapezoid.neighbors["bottom_left"] = Z_trapezoid

def set_neighbors_case_one_point_inside(X_trapezoid, Y_trapezoid, Z_trapezoid):
    # Set neighbors for X, Y, Z trapezoids
    X_trapezoid.neighbors["top_right"] = Y_trapezoid
    X_trapezoid.neighbors["bottom_right"] = Z_trapezoid

    Y_trapezoid.neighbors["top_left"] = X_trapezoid

    Z_trapezoid.neighbors["bottom_left"] = X_trapezoid

#****************************************************************************************************

def random_incremental_algorithm(segments, bounding_box, trapezoids):
    global root_node
    left_point = Point(bounding_box[0], bounding_box[1], is_left=True)
    right_point = Point(bounding_box[2], bounding_box[3], is_left=False)

    initial_trapezoid = Trapezoid(None, None, left_point, right_point)
    root_node = LeafNode(initial_trapezoid.trapezoid_id)
    trapezoids[initial_trapezoid.trapezoid_id] = [f"T{root_node.trapezoid_id}", root_node, initial_trapezoid]
    D[initial_trapezoid.identifier].append(root_node)
    display_dictionaries()
    for segment_coords in segments:
        segment = Segment(Point(segment_coords[0], segment_coords[1], is_left=True),
                          Point(segment_coords[2], segment_coords[3], is_left=False))
        print(segment)
        intersected_trapezoids = find_intersected_trapezoids(segment, trapezoids)
        print("Intersected trapezoids:", intersected_trapezoids)
        update(intersected_trapezoids, trapezoids, segment.left, segment.right, segment)






def create_adjacency_matrix_and_output(filename, trapezoids):
    # Gather and sort all unique keys from dictionaries
    combined_keys = sorted({*A, *B, *C, *D})
    df = pd.DataFrame(0, index=combined_keys, columns=combined_keys)

    def get_node_id(node):
        if isinstance(node, XNode):
            return node.point_represented.identifier
        elif isinstance(node, YNode):
            return node.segment.identifier
        elif isinstance(node, LeafNode):
            return trapezoids[node.trapezoid_id][0]
        return node

    def process_nodes(key, nodes, above_below=False):
        for node in nodes:
            left = get_node_id(node.left if not above_below else node.above)
            right = get_node_id(node.right if not above_below else node.below)
            df.loc[[left, right], key] += 1

    # Populate the DataFrame for each dictionary
    for key in combined_keys:
        if key in A or key in B:
            process_nodes(key, A.get(key, []))
        if key in C:
            process_nodes(key, C[key], above_below=True)

    # Add total sums as the last row and column
    df['Total'] = df.sum(axis=1)
    df.loc['Total'] = df.sum()

    # Write the DataFrame to a text file with aligned formatting
    with open(filename, 'w') as f:
        f.write(df.to_string(index=True, header=True, justify='center'))




def traverse_x_node(current_node, query_point):
    if query_point.x <= current_node.x_value:
        return current_node.right, current_node.point_represented.identifier
    return current_node.left, current_node.point_represented.identifier

def traverse_y_node(current_node, query_point):

    segment = current_node.segment
    if query_point_lies_on_segment(query_point, segment):
        return current_node.above, segment.identifier
    elif is_point_above_segment(query_point, segment):
        return current_node.above, segment.identifier
    else:
        return current_node.below, segment.identifier

def query_points(root_node, query_point):
    path, current_node = [], root_node

    while not isinstance(current_node, LeafNode):
        current_node, node_id = (traverse_x_node(current_node, query_point)
                                 if isinstance(current_node, XNode)
                                 else traverse_y_node(current_node, query_point))
        path.append(node_id)

    return path + [f"T{current_node.trapezoid_id}"]




def initialize_trapezoidal_map(input_file):
    segments, bounding_box = process(input_file)
    print("Segments:", segments)
    print("Bounding Box:", bounding_box)
    return segments, bounding_box

def build_trapezoidal_map(segments, bounding_box):
    trapezoids = collections.defaultdict(list)
    random_incremental_algorithm(segments, bounding_box, trapezoids)
    return trapezoids

def write_adjacency_matrix(output_file, trapezoids):
    print("Output adjacency matrix...")
    create_adjacency_matrix_and_output(output_file, trapezoids)

def get_query_point():
    input_values = input("Please enter the point coordinates (x y): ").split()
    query_point = Point(float(input_values[0]), float(input_values[1]))
    return query_point

def main():
    segments, bounding_box = initialize_trapezoidal_map('InputFiles/ya2390.txt')
    trapezoids = build_trapezoidal_map(segments, bounding_box)
    display_dictionaries()
    print("-" * 95)
    write_adjacency_matrix('output.txt', trapezoids)
    print("-" * 95)

    query_point = get_query_point()
    path = query_points(root_node, query_point)
    print("Traversal path: " + ' '.join(path))


if __name__ == '__main__':
    main()
