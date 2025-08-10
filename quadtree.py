# quadtree.py
import math

class Point:
    """Represents a point at (x, y) with optional data (e.g., driver id)."""
    def __init__(self, x, y, data=None):
        self.x = float(x)
        self.y = float(y)
        self.data = data

    def __repr__(self):
        return f"Point({self.x:.2f}, {self.y:.2f}, data={self.data})"


class Rectangle:
    """Axis-aligned rectangle with top-left (x, y) and size (width, height)."""
    def __init__(self, x, y, width, height):
        self.x = float(x)
        self.y = float(y)
        self.width  = float(width)
        self.height = float(height)

    def contains(self, point):
        """Return True if the point lies within this rectangle (inclusive-left/top)."""
        return (self.x <= point.x < self.x + self.width and
                self.y <= point.y < self.y + self.height)

    def distance_sq_to_point(self, point):
        """
        Squared shortest distance from a point to this rectangle (0 if inside).
        Used for pruning in nearest-neighbor search.
        """
        dx = 0.0
        if point.x < self.x:
            dx = self.x - point.x
        elif point.x > self.x + self.width:
            dx = point.x - (self.x + self.width)

        dy = 0.0
        if point.y < self.y:
            dy = self.y - point.y
        elif point.y > self.y + self.height:
            dy = point.y - (self.y + self.height)

        return dx*dx + dy*dy

    def __repr__(self):
        return f"Rect(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


class QuadtreeNode:
    """
    One node/region in the Quadtree.
    Holds up to `capacity` points; if exceeded, subdivides into 4 children.
    """
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.capacity = int(capacity)
        self.points = []
        self.divided = False
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

    def subdivide(self):
        """Split this node into four child nodes (NW, NE, SW, SE)."""
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.width, self.boundary.height
        hw = w / 2.0
        hh = h / 2.0

        nw = Rectangle(x,       y,       hw, hh)
        ne = Rectangle(x + hw,  y,       hw, hh)
        sw = Rectangle(x,       y + hh,  hw, hh)
        se = Rectangle(x + hw,  y + hh,  hw, hh)

        self.northwest = QuadtreeNode(nw, self.capacity)
        self.northeast = QuadtreeNode(ne, self.capacity)
        self.southwest = QuadtreeNode(sw, self.capacity)
        self.southeast = QuadtreeNode(se, self.capacity)
        self.divided = True

        # Re-insert any existing points into children
        old_points = self.points
        self.points = []
        for p in old_points:
            self._insert_into_children(p)

    def _insert_into_children(self, point):
        """Helper: insert into any one child whose boundary contains the point."""
        return (self.northwest.insert(point) or
                self.northeast.insert(point) or
                self.southwest.insert(point) or
                self.southeast.insert(point))

    def insert(self, point):
        """Insert a point into this node (or its children)."""
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True

        if not self.divided:
            self.subdivide()

        return self._insert_into_children(point)

    def find_nearest(self, query_point, best_point=None, best_dist_sq=float('inf')):
        """
        Recursive nearest-neighbor search with pruning.
        Returns (best_point, best_dist_sq).
        """
        # If this region can't have a closer point, prune it.
        if self.boundary.distance_sq_to_point(query_point) > best_dist_sq:
            return best_point, best_dist_sq

        # Check points in this node
        for p in self.points:
            d2 = (p.x - query_point.x)**2 + (p.y - query_point.y)**2
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_point = p

        # Recurse into children (closest-first by boundary distance)
        if self.divided:
            children = [
                self.northwest, self.northeast,
                self.southwest, self.southeast
            ]
            children.sort(key=lambda child: child.boundary.distance_sq_to_point(query_point))
            for child in children:
                best_point, best_dist_sq = child.find_nearest(query_point, best_point, best_dist_sq)

        return best_point, best_dist_sq


class Quadtree:
    """User-facing Quadtree wrapper that manages the root node."""
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.root = QuadtreeNode(boundary, capacity)

    def insert(self, point):
        return self.root.insert(point)

    def find_nearest(self, query_point):
        best_point, best_dist_sq = self.root.find_nearest(query_point)
        return best_point, math.sqrt(best_dist_sq) if best_point is not None else (None, float('inf'))

    def __repr__(self):
        return f"Quadtree(boundary={self.boundary}, capacity={self.root.capacity})"