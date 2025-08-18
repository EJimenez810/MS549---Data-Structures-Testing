# simulation.py
import heapq

# Import the Graph class
try:
    from graph import Graph
except Exception:
    Graph = None # A failsafe in the event graph.py is not present.

# Travel speed placeholder
# travel_time = manhattan_distance + TRAVEL_SPEED_FACTOR
TRAVEL_SPEED_FACTOR = 1.0 # seconds per grid unit

# Defines the Simulation class  
class Simulation:
    """
    Manages the simulation including cars, riders, and the city map.
    Discrete-Event Simulation engine (prototype).
    Stores events in a mini-heap of (timestamp, event_type, data) and processes them in chronological order
    """

    def __init__(self, map_filename=None):
        # Core State
        self.current_time = 0.0

        # min-heap of (time, type, data)
        self.events = []
        
        # Dictionary to store Cars by ID      
        self.cars = {}
        
        # Dictionary to store Riders by ID
        self.riders = {}

        # City map
        self.map = None
        if map_filename and Graph is not None:
            try:
                # Load map from CSV file
                self.map = Graph()
                if hasattr(self.map, "load_from_file"):
                    self.map.load_from_file(map_filename)
            except Exception:
                # Making map optional for this prototype; ignoring load errors
                self.map = None

    def __str__(self):
        # Only include map if it exists and not None
        # Avoid printing "None" when no map is loaded
        map_str = f"\n{self.map}" if self.map is not None else ""
        # Print a summary of how many cars and riders are in the simulation
        return f"Simulation with {len(self.cars)} cars and {len(self.riders)} riders.{map_str}"

    # Log
    def log(self, msg: str):
        print(f"TIME {self.current_time:6.2f}: {msg}")

    # Event Scheduling
    def add_event_at(self, at_time: float, event_type: str, data):
        heapq.heappush(self.events, (at_time, event_type, data))
        # Audit log prints immediately to see scheduling
        print(f"TIME {self.current_time:6.2f}: SCHEDULE -> {event_type} at {at_time:6.2f} data={data}")

    def add_event_after(self, delay: float, event_type: str, data):
        self.add_event_at(self.current_time + delay, event_type,data)

    # Navigation placeholder
    @staticmethod
    def manhattan_distance(a,b):
        ax, ay = a
        bx, by = b
        return abs(ax - bx) + abs(ay - by)

    def calculate_travel_time(self, start_location, end_location):
        distance = self.manhattan_distance(start_location, end_location)
        return distance * TRAVEL_SPEED_FACTOR

    # Matching placeholder
    def find_closest_car_brute_force(self, rider_location):
        best_car = None
        best_dist = float('inf')
        for car in self.cars.values():
            # Only considers available cars
            if getattr(car, 'status', 'available') == 'available':
                d = self.manhattan_distance(car.location, rider_location)
                if d < best_dist:
                    best_dist = d
                    best_car = car
        return best_car
    
    # Event handlers
    def handle_rider_request(self, rider):
        # Accept either a Rider or a rider_id
        if isinstance(rider, str):
            rider = self.riders[rider]
        self.log(f"RIDER_REQUEST({rider.id}) at {rider.start_location}")

        car = self.find_closest_car_brute_force(rider.start_location)
        if not car:
            self.log(f"No available cars for {rider.id}. (Prototype: skipping)")
            return

        # Link car to rider and set status
        # (If Car doesn't define 'assigned_rider', we can still attach it dynamically.)
        setattr(car, 'assigned_rider', rider)
        car.status = 'en_route_to_pickup'
        self.log(f"DISPATCH -> Car {car.id} to {rider.id}. Car status={car.status}")

        # Schedule pickup arrival
        pickup_time = self.calculate_travel_time(car.location, rider.start_location)
        eta = self.current_time + pickup_time
        self.add_event_at(eta, 'ARRIVAL', car)

    def handle_arrival(self, car):
        # Arrival means either pickup finished or dropoff finished, depending on car.status
        status = getattr(car, 'status', 'available')
        rider = getattr(car, 'assigned_rider', None)

        if status == 'en_route_to_pickup' and rider is not None:
            # Perform pickup
            car.location = rider.start_location
            car.status = 'en_route_to_destination'
            rider.status = 'in_car'
            self.log(
                f"PICKUP -> Car {car.id} picked up {rider.id} at {car.location}. "
                f"Car status={car.status}, Rider status={rider.status}"
            )

            # Schedule dropoff arrival
            drop_time = self.calculate_travel_time(car.location, rider.destination)
            eta = self.current_time + drop_time
            self.add_event_at(eta, 'ARRIVAL', car)

        elif status == 'en_route_to_destination' and rider is not None:
            # Perform dropoff
            car.location = rider.destination
            rider.status = 'completed'
            car.status = 'available'
            car.assigned_rider = None
            self.log(
                f"DROPOFF -> Car {car.id} dropped off {rider.id} at {car.location}. "
                f"Car status={car.status}, Rider status={rider.status}"
            )

        else:
            # In a fuller engine log unexpected states
            self.log(f"ARRIVAL with unexpected car status={status} rider={getattr(rider,'id',None)} (no-op)")

    # Main Loop
    def run(self):
        self.log("SIMULATION START")
        while self.events:
            next_time, event_type, data = heapq.heappop(self.events)
            self.current_time = next_time

            if event_type == 'RIDER_REQUEST':
                # data is a Rider or rider_id
                self.handle_rider_request(data)
            elif event_type == 'ARRIVAL':
                # data is a Car
                self.handle_arrival(data)
            else:
                self.log(f"Unknown event type: {event_type}")
        self.log("SIMULATION END")
        