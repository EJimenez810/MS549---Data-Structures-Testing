# car.py

from pathfinding import find_shortest_path

class Car:
    """
    Represents a single car.
    """

    def __init__(self, car_id, initial_location):
        # car_id uniquely identifies the car (ex: "CAR001")
        # initial_location is a tuple representing starting point (x, y) in final milestone
        self.id = car_id
        self.location = initial_location  # (x, y)

        # Car starts as available
        self.status = "available"

        # Destination is not set yet
        self.destination = None

        # Rider currently assigned to this car, None if available
        self.assigned_rider = None

        # Route-related attributes for Dijkstra
        self.route = []
        self.route_time = 0

    # NEW: helper to snap (x, y) to nearest graph vertex id
    @staticmethod
    def _nearest_vertex_id(point, graph):
        """
        Accepts either:
          - a node id string (returns it unchanged), or
          - an (x, y) tuple (returns nearest node id by Euclidean distance).
        """
        # If it's already a node id, just return it
        if isinstance(point, str):
            return point

        x, y = point
        best_id = None
        best_d2 = float("inf")
        for node_id, (nx, ny) in graph.node_coordinates.items():
            d2 = (x - nx) ** 2 + (y - ny) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_id = node_id
        return best_id

    def calculate_route(self, destination, graph):
        """
        Uses Dijkstra's algorithm to calculate a route from current location to destination.
        Supports (x, y) inputs by snapping to nearest graph nodes.
        Updates self.route and self.route_time.

        Args:
            destination: str node id OR (x, y) tuple
            graph: Graph object with .adjacency_list and .node_coordinates
        """
        # Snap current car position and destination to nearest node ids (if needed)
        start_id = self._nearest_vertex_id(self.location, graph)
        end_id   = self._nearest_vertex_id(destination, graph)

        # Run Dijkstra over node ids
        path, total_time = find_shortest_path(graph, start_id, end_id)
        if path:
            self.route = path
            self.route_time = float(total_time)
            print(f"Car {self.id} route calculated: {self.route} with total time {self.route_time}")
        else:
            self.route = []
            self.route_time = 0
            print(f"Car {self.id}: No route found from {start_id} to {end_id}.")

    def __str__(self):
        # returns car's current status and assigned rider
        rider_info = f" -> Rider {self.assigned_rider.id}" if self.assigned_rider else ""
        return f"Car {self.id} at {self.location} - Status: {self.status}{rider_info}"