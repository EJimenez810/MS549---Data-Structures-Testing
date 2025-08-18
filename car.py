# car.py

# Import the pathfinding function
from pathfinding import find_shortest_path 

# Defines the Car Class in the ride-sharing system
class Car:
    """
    Represents a single car.
    """

    def __init__(self, car_id, initial_location):
        # car_id uniquely identifies the car (ex: "CAR001)
        # initial_location is a tuple representing starting point
        self.id = car_id
        self.location = initial_location

        # Car starts as available
        self.status = "available"

        # Destination is not set yet
        self.destination = None

        # Rider currently assigned to this car, None if available
        self.assigned_rider = None

        # Route-related attributes for Dijkstra
        self.route = []
        self.route_time = 0

    def calculate_route(self, destination, graph):
        """
        Uses Dijkstra's algorithm to calculate route from current location to destination.
        Updates self.route and self.route_time.
        """
        path, total_time = find_shortest_path(graph, self.location, destination)
        if path:
            self.route = path
            self.route_time = total_time
            print(f"Car {self.id} route calculated: {self.route} with total time {self.route_time}")
        else:
            print(f"Car {self.id}: No route found from {self.location} to {destination}.")

    def __str__(self):
        # returns cars current status and assigned rider
        rider_info = f" -> Rider {self.assigned_rider.id}" if self.assigned_rider else ""
        return f"Car {self.id} at {self.location} - Status: {self.status}{rider_info}"
