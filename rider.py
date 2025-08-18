# rider.py

# Defines the Rider class
class Rider:
    """
    Represents a rider/customer in the simulation
    """

    def __init__(self, rider_id, start_location, destination):
        # rider_id uniquely identifies a rider (ex: "RIDER_A")
        self.id = rider_id
        # start_location is where the rider will be picked up
        self.start_location = start_location
        # destination is where the rider wants to go
        self.destination = destination
        # Rider begins in a waiting state
        self.status = "waiting"

        # Runtime Check
        if not (isinstance(self.start_location, tuple) and len(self.start_location) == 2):
            raise ValueError("start_location must be a 2-tuple like (x, y)")
        if not (isinstance(self.destination, tuple) and len(self.destination) == 2):
            raise ValueError("destination must be a 2-tuple like (x, y)")

    def __str__(self):
        # Returns the rider's status and destination
       return (f"Rider {self.id} at {self.start_location} "
                f"status={self.status} -> {self.destination}")
