# simulation.py

# The following is for final report documentation:
    # Final milestone simulation engine requirements:
        # Discrete-event core (heapq)
        # Dynamic rider generation (expovariate arrivals)
        # True travel time via Dijkstra on 7-col map (snap (x,y) -> nearest graph node)
        # k-nearest matching with Quadtree (AUTO fallback to brute-force if Point lacks `payload`)
        # Instrumentation (trip_log) -> KPIs -> single PNG visualization
        # Smart car seeding (auto/graph/box/manual) via CLI args

    # If running with legacy 3-col CSV or no map, gracefully fall back to:
        # Manhattan-based travel time
        # Box seeding (0..100 world)
        # Brute-force matching (still correct, just less efficient)

import heapq
import argparse
import random
import math
import matplotlib.pyplot as plt

# Try to import Graph (supports 3-col and 7-col)
try:
    from graph import Graph
except Exception:
    Graph = None  # Run without a map (placeholders kick in)

# Import Car and Dijkstra
from car import Car
from pathfinding import find_shortest_path  # expected: (path_list, total_cost)

# Try to import Quadtree; if not present or incompatible, auto-fallback
try:
    from quadtree import Quadtree, Rectangle, Point
except Exception:
    Quadtree = Rectangle = Point = None  # disable quadtree usage

# Global constants (tunable)

TRAVEL_SPEED_FACTOR = 1.0       # Used only when 7-col map not loaded (Manhattan model)
DEFAULT_MEAN_ARRIVAL = 30.0     # Mean inter-arrival time for riders (expovariate)
RANDOM_SEED = 42                # Deterministic runs for grading reproducibility
K_NEAREST = 5                   # Quadtree: candidate cars to evaluate via Dijkstra


class Simulation:
    """
    Event-driven ride-sharing simulation with:
        Real routing via Dijkstra (when 7-col map is present)
        Quadtree k-NN matching (falls back to brute-force when quadtree unsupported)
        Dynamic rider request generation and daisy-chained events
        Instrumentation for KPIs and visualization
    """

    def __init__(self, map_filename=None, max_time=2000.0, mean_arrival=DEFAULT_MEAN_ARRIVAL, seed=RANDOM_SEED):
        # Core Sim State
        self.current_time = 0.0                    # simulation clock
        self.max_time = float(max_time)            # horizon / stopping condition
        self.events = []                           # min-heap of (time, event_type, data)

        # Entities
        self.cars = {}                             # car_id -> Car
        self.riders = {}                           # rider_id -> Rider
        self._rider_counter = 0                    # for unique rider IDs

        # Instrumentation
        self.trip_log = []                         # each completed trip -> dict of timestamps/metrics

        # Arrival Process
        self.mean_arrival = float(mean_arrival)    # average gap between rider requests
        self.rng = random.Random(seed)             # deterministic RNG

        # Map / Graph
        self.map = None
        if map_filename and Graph is not None:
            try:
                self.map = Graph()
                # Prefer 7-col loader if present; updated Graph has load_map_data()
                if hasattr(self.map, "load_map_data"):
                    self.map.load_map_data(map_filename)
                elif hasattr(self.map, "load_from_file"):
                    self.map.load_from_file(map_filename)
            except Exception:
                # Allows map to be absent; placeholders will take over
                self.map = None

        # Quadtree
        # Will detect if Point supports a `payload` parameter and enables accordingly.
        self.qtree = None
        self._qtree_payload_supported = False  # True if attaching Car objects to quadtree Points
        self._qtree_enabled = False            # True if using quadtree for matching
        self._init_quadtree_if_possible()

    def __str__(self):
        # Summary used in debugging
        map_str = f"\n{self.map}" if self.map is not None else ""
        return f"Simulation with {len(self.cars)} cars and {len(self.riders)} riders.{map_str}"

    # Logging helper
    
    def log(self, msg: str):
        # Standardized log prefix with current time
        print(f"TIME {self.current_time:6.2f}: {msg}")

    # Event scheduling helpers
    
    def _schedule_at(self, when: float, event_type: str, data):
        """Schedule an event at an absolute timestamp."""
        heapq.heappush(self.events, (when, event_type, data))
        # This is for code-review to show the sequencing
        print(f"TIME {self.current_time:6.2f}: SCHEDULE -> {event_type} at {when:6.2f} data={data}")

    def _schedule_next_request(self):
        """Daisy-chain another rider request using expovariate inter-arrival time."""
        delta = self.rng.expovariate(1.0 / self.mean_arrival) if self.mean_arrival > 0 else 0.0
        next_t = self.current_time + delta
        if next_t <= self.max_time:
            self._schedule_at(next_t, 'RIDER_REQUEST', None)

    # Dynamic rider generation
    
    def generate_rider_request(self):
        """
        Create a new Rider with random (x,y) start/destination.
        If the 7-col map is present, use its bounding box; else default to [0,100]^2.
        """
        from rider import Rider  # local import to avoid circular dependency
        self._rider_counter += 1
        rid = f"RIDER_{self._rider_counter:05d}"

        # Determine spatial bounds
        if self.map and hasattr(self.map, "node_coordinates") and self.map.node_coordinates:
            xs = [xy[0] for xy in self.map.node_coordinates.values()]
            ys = [xy[1] for xy in self.map.node_coordinates.values()]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
        else:
            minx = miny = 0.0
            maxx = maxy = 100.0

        # Sample random start and destination
        sx = self.rng.uniform(minx, maxx)
        sy = self.rng.uniform(miny, maxy)
        dx = self.rng.uniform(minx, maxx)
        dy = self.rng.uniform(miny, maxy)

        rider = Rider(rid, (sx, sy), (dx, dy))
        self.riders[rider.id] = rider
        return rider

    # Routing / travel-time layer

    @staticmethod
    def manhattan_distance(a, b):
        """Grid distance ignoring diagonals; used only when 7-col map is unavailable."""
        ax, ay = a
        bx, by = b
        return abs(ax - bx) + abs(ay - by)

    def calculate_travel_time_placeholder(self, start_location, end_location):
        """Fallback travel-time model: Manhattan * constant."""
        return self.manhattan_distance(start_location, end_location) * TRAVEL_SPEED_FACTOR

    def _nearest_vertex_id(self, point_xy):
        """
        Snap an (x,y) to the nearest graph node by Euclidean distance.
        Returns node_id or None if coordinates are unavailable.
        """
        if not (self.map and hasattr(self.map, "node_coordinates") and self.map.node_coordinates):
            return None
        px, py = point_xy
        best_id, best_d2 = None, float("inf")
        for node_id, (nx, ny) in self.map.node_coordinates.items():
            d2 = (px - nx) ** 2 + (py - ny) ** 2
            if d2 < best_d2:
                best_d2, best_id = d2, node_id
        return best_id

    def true_travel_time(self, start_xy, end_xy):
        """
        Preferred travel-time model:
            Snap (x,y) -> nearest graph nodes
            Run Dijkstra (find_shortest_path) to get real shortest-path cost
        Falls back to Manhattan model if we don't have a coordinate-enabled map.
        """
        if not (self.map and hasattr(self.map, "node_coordinates") and self.map.node_coordinates):
            return self.calculate_travel_time_placeholder(start_xy, end_xy)

        s = self._nearest_vertex_id(start_xy)
        t = self._nearest_vertex_id(end_xy)
        if s is None or t is None:
            return float("inf")  # path not defined (shouldn't happen on consistent maps)

        path, total = find_shortest_path(self.map, s, t)  # your function; must return (path, cost)
        return float(total) if total is not None else float("inf")

    # Quadtree integration (optional)
    
    def _init_quadtree_if_possible(self):
        """
        Initialize quadtree with bounds derived from the map (if present) or a default 0..100 box.
        Detect whether your Point supports a `payload` parameter. If not, we disable quadtree usage
        and automatically fall back to brute-force (still correct for small maps).
        """
        # Start disabled by default
        self.qtree = None
        self._qtree_payload_supported = False
        self._qtree_enabled = False

        # If your quadtree module isn't importable, leave disabled
        if Quadtree is None or Rectangle is None or Point is None:
            return

        # Compute spatial bounds
        if self.map and hasattr(self.map, "node_coordinates") and self.map.node_coordinates:
            xs = [xy[0] for xy in self.map.node_coordinates.values()]
            ys = [xy[1] for xy in self.map.node_coordinates.values()]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
        else:
            minx = miny = 0.0
            maxx = maxy = 100.0

        # Convert to center/half-width rectangle (typical quadtree API)
        half_w = max(1.0, (maxx - minx) / 2.0)
        half_h = max(1.0, (maxy - miny) / 2.0)
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0

        # Try to construct the quadtree
        try:
            self.qtree = Quadtree(Rectangle(cx, cy, half_w, half_h), capacity=4)
        except Exception:
            self.qtree = None
            return

        # Probe for `payload=` support on Point; if unsupported, disable tree usage
        try:
            _ = Point(0.0, 0.0, payload="probe")   # TypeError if payload not accepted
            self._qtree_payload_supported = True
        except TypeError:
            self._qtree_payload_supported = False
        except Exception:
            self._qtree_payload_supported = False

        # Only enable quadtree if we can attach Car objects to points
        self._qtree_enabled = self._qtree_payload_supported

    def qtree_insert_car(self, car: Car):
        """Insert an available car into the quadtree index (if enabled)."""
        if not (self.qtree and self._qtree_enabled):
            return
        self.qtree.insert(Point(car.location[0], car.location[1], payload=car))

    def qtree_remove_car(self, car: Car):
        """Remove a car from the quadtree index (if enabled)."""
        if not (self.qtree and self._qtree_enabled):
            return
        try:
            self.qtree.remove(Point(car.location[0], car.location[1], payload=car))
        except Exception:
            # Some quadtree implementations require the *same* Point instance.
            # Rebuild as a safe fallback to avoid "dangling" points.
            self._rebuild_qtree()

    def _rebuild_qtree(self):
        """Rebuild index from currently available cars (safety net when remove() fails)."""
        if not (self.qtree and self._qtree_enabled):
            return
        # snapshot current cars
        available = [c for c in self.cars.values() if getattr(c, "status", "available") == "available"]
        # re-init tree with same bounds
        self._init_quadtree_if_possible()
        # reinsert
        for c in available:
            self.qtree_insert_car(c)

    def qtree_k_candidates(self, xy, k=K_NEAREST):
        """
        Return up to k nearest available cars (by spatial distance) via quadtree.
        If tree is disabled/unavailable, return [] so the caller falls back to brute-force.
        """
        if not (self.qtree and self._qtree_enabled):
            return []
        try:
            raw = self.qtree.find_k_nearest((xy[0], xy[1]), k)  # adapt to your quadtree API if needed
        except Exception:
            self._rebuild_qtree()
            return []
        cars = []
        for item in raw:
            # Expect Points that carry a Car object in `payload`
            if hasattr(item, "payload") and isinstance(item.payload, Car):
                cars.append(item.payload)
        return cars

    # Matching strategy (k-NN + Dijkstra) with graceful fallback
    
    def find_best_car(self, rider_location):
        """
        Choose the *available* car with the smallest Dijkstra time to the rider.
        If quadtree is enabled: evaluate only k nearest *spatial* candidates.
        Else: brute-force all cars (still correct for small fleets).
        Returns: (best_car, best_time_to_pickup)
        """
        # Try quadtree k-NN first
        candidates = []
        if self.qtree and self._qtree_enabled:
            candidates = [c for c in self.qtree_k_candidates(rider_location, K_NEAREST)
                          if getattr(c, "status", "available") == "available"]

        # If no candidates (tree disabled or empty), brute-force all available cars
        if not candidates:
            candidates = [c for c in self.cars.values() if getattr(c, "status", "available") == "available"]

        # Score candidates by *true* time-to-pickup (Dijkstra)
        best_car, best_time = None, float("inf")
        for car in candidates:
            t = self.true_travel_time(car.location, rider_location)
            if t < best_time:
                best_time, best_car = t, car
        return best_car, best_time

    # Event handlers
    
    def handle_rider_request(self, rider):
        """
        RIDER_REQUEST:
            If rider is None: generate a new rider (dynamic arrivals)
            Log request time
            Schedule the *next* rider request (daisy-chain)
            Choose best car by Dijkstra (k-NN if quadtree enabled, else brute-force)
            Schedule PICKUP_ARRIVAL
        """
        # Generate or fetch rider
        if rider is None:
            rider = self.generate_rider_request()
        elif isinstance(rider, str):
            rider = self.riders[rider]

        # Instrumentation: mark request time
        rider.request_time = self.current_time
        self.log(f"RIDER_REQUEST({rider.id}) at {rider.start_location}")

        # Daisy-chain to keep arrivals coming
        self._schedule_next_request()

        # Choose car
        car, best_time = self.find_best_car(rider.start_location)
        if not car or best_time == float("inf"):
            self.log(f"No available cars with a connected path for {rider.id}.")
            return

        # Link + status transition
        car.assigned_rider = rider
        car.status = 'en_route_to_pickup'
        self.log(f"DISPATCH -> Car {car.id} to {rider.id} (eta {best_time:.2f})")

        # Remove from "available set" in quadtree (if enabled)
        self.qtree_remove_car(car)

        # Schedule pickup
        self._schedule_at(self.current_time + best_time, 'PICKUP_ARRIVAL', car)

    def handle_pickup_arrival(self, car):
        """
        PICKUP_ARRIVAL:
            Car reaches rider.start_location
            Update locations and statuses
            Record rider.pickup_time
            Schedule DROPOFF_ARRIVAL using true Dijkstra time to destination
        """
        rider = car.assigned_rider
        if not rider:
            self.log(f"PICKUP_ARRIVAL for {car.id} but no rider assigned; ignoring.")
            return

        # State updates
        car.location = rider.start_location
        car.status = 'en_route_to_destination'
        rider.status = 'in_car'
        rider.pickup_time = self.current_time
        self.log(f"PICKUP -> {car.id} picked up {rider.id} at {car.location}")

        # Schedule dropoff
        drop_t = self.true_travel_time(car.location, rider.destination)
        self._schedule_at(self.current_time + drop_t, 'DROPOFF_ARRIVAL', car)

    def handle_dropoff_arrival(self, car):
        """
        DROPOFF_ARRIVAL:
            Car reaches rider.destination
            Record rider.dropoff_time; mark trip complete
            Log trip (append to trip_log)
            Make car available again and reinsert to quadtree (if enabled)
        """
        rider = car.assigned_rider
        if not rider:
            self.log(f"DROPOFF_ARRIVAL for {car.id} but no rider assigned; ignoring.")
            return

        # State updates
        car.location = rider.destination
        rider.status = 'completed'
        rider.dropoff_time = self.current_time

        # Log completed trip for analysis
        self.log_trip_data(rider)

        # Reset car, make it available for matching again
        car.assigned_rider = None
        car.status = 'available'
        self.qtree_insert_car(car)
        self.log(f"DROPOFF -> {car.id} dropped off {rider.id} at {car.location}; car available")

    # Instrumentation & KPIs

    def log_trip_data(self, rider):
        """Append a completed trip record (with derived metrics) to trip_log."""
        wait = (rider.pickup_time - rider.request_time) if (rider.pickup_time is not None and rider.request_time is not None) else None
        dur  = (rider.dropoff_time - rider.pickup_time) if (rider.dropoff_time is not None and rider.pickup_time is not None) else None
        record = {
            "rider_id": rider.id,
            "request_time": rider.request_time,
            "pickup_time": rider.pickup_time,
            "dropoff_time": rider.dropoff_time,
            "wait_time": wait,
            "trip_duration": dur,
        }
        self.trip_log.append(record)
        self.log(f"Trip logged -> {rider.id} (wait={wait if wait is not None else 'NA'} dur={dur if dur is not None else 'NA'})")

    def analyze_results(self):
        """
        Aggregate trip_log into the 3 required KPIs:
            Average Rider Wait Time
            Average Trip Duration
            Driver Utilization (% of time cars are busy)
        """
        if not self.trip_log:
            print("\n--- Simulation Analysis ---\nNo trips were completed.\n")
            return {
                "completed_trips": 0,
                "average_wait_time": 0.0,
                "average_trip_duration": 0.0,
                "driver_utilization_percent": 0.0,
            }

        waits = [t["wait_time"] for t in self.trip_log if t["wait_time"] is not None]
        durs  = [t["trip_duration"] for t in self.trip_log if t["trip_duration"] is not None]
        n = len(self.trip_log)
        avg_wait = (sum(waits) / n) if n else 0.0
        avg_dur  = (sum(durs) / n) if n else 0.0

        # Utilization: sum of trip durations / (num_cars * total_sim_time)
        total_time_on_trips = sum(durs)
        num_cars = max(len(self.cars), 1)
        denom = num_cars * max(self.current_time, 1e-9)
        utilization = (total_time_on_trips / denom) * 100.0

        results = {
            "completed_trips": n,
            "average_wait_time": avg_wait,
            "average_trip_duration": avg_dur,
            "driver_utilization_percent": utilization,
        }

        # Console summary
        print("\n--- Simulation Analysis ---")
        print(f"Completed Trips: {n}")
        print(f"Average Rider Wait Time: {avg_wait:.2f}")
        print(f"Average Trip Duration:  {avg_dur:.2f}")
        print(f"Driver Utilization:     {utilization:.2f}%")
        print("--------------------------\n")

        return results

    def render_summary_png(self, filename: str = "simulation_summary.png"):
        """
        Produce the single integrated visualization:
            Left: city nodes (if available) + final car positions
            Right: KPI text + wait-time histogram
        """
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.25)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])

        # Plot map nodes if we have coordinates (7-col map)
        if self.map and hasattr(self.map, "node_coordinates") and self.map.node_coordinates:
            xs = [xy[0] for xy in self.map.node_coordinates.values()]
            ys = [xy[1] for xy in self.map.node_coordinates.values()]
            ax_map.scatter(xs, ys, s=5, alpha=0.3)

        ax_map.set_title("City Map & Final Car Positions")
        ax_map.set_xlabel("x")
        ax_map.set_ylabel("y")

        # Plot final car locations
        if self.cars:
            cx = [c.location[0] for c in self.cars.values()]
            cy = [c.location[1] for c in self.cars.values()]
            ax_map.scatter(cx, cy, s=18)

        # KPI pane
        res = self.analyze_results()
        ax_info.axis("off")
        y0, dy = 0.95, 0.08
        ax_info.text(0.05, y0, "Performance Summary", fontsize=12, fontweight="bold", transform=ax_info.transAxes)
        ax_info.text(0.05, y0 - dy*1, f"Completed Trips: {res['completed_trips']}", transform=ax_info.transAxes)
        ax_info.text(0.05, y0 - dy*2, f"Avg Wait: {res['average_wait_time']:.2f}", transform=ax_info.transAxes)
        ax_info.text(0.05, y0 - dy*3, f"Avg Duration: {res['average_trip_duration']:.2f}", transform=ax_info.transAxes)
        ax_info.text(0.05, y0 - dy*4, f"Utilization: {res['driver_utilization_percent']:.2f}%", transform=ax_info.transAxes)

        # Wait-time histogram
        waits = [t["wait_time"] for t in self.trip_log if t["wait_time"] is not None]
        if waits:
            inset = fig.add_axes([0.68, 0.18, 0.27, 0.23])  # [left, bottom, width, height] in figure coords
            inset.hist(waits, bins=min(15, max(5, int(math.sqrt(len(waits))))))
            inset.set_xlabel("wait")
            inset.set_ylabel("count")
            ax_info.text(0.05, y0 - dy*5.5, "Wait Time Histogram", fontweight="bold", transform=ax_info.transAxes)

        fig.savefig(filename, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization -> {filename}")

    # Main event loop

    def run(self):
        """Pop events in chronological order until the queue is empty or we hit max_time."""
        self.log("SIMULATION START")

        # Kick off first dynamic rider request at t=0
        self._schedule_at(0.0, 'RIDER_REQUEST', None)

        # Core loop: pop from min-heap, update clock, dispatch to handler
        while self.events and self.current_time <= self.max_time:
            t, etype, data = heapq.heappop(self.events)
            self.current_time = t

            if etype == 'RIDER_REQUEST':
                self.handle_rider_request(data)
            elif etype == 'PICKUP_ARRIVAL':
                self.handle_pickup_arrival(data)
            elif etype == 'DROPOFF_ARRIVAL':
                self.handle_dropoff_arrival(data)
            else:
                self.log(f"Unknown event: {etype}")

        self.log("SIMULATION END")

# Car seeding helpers (support auto/graph/box/manual from CLI)

def seed_cars_manual(sim: Simulation, manual_list):
    """
    Place cars exactly at user-provided positions (used when --seed-mode manual).
    manual_list: [(x1,y1), (x2,y2), ...]
    """
    sim.cars.clear()
    for i, (x, y) in enumerate(manual_list, start=1):
        cid = f"CAR{i:03d}"
        sim.cars[cid] = Car(cid, (float(x), float(y)))
        sim.qtree_insert_car(sim.cars[cid])  # index only if quadtree enabled

def seed_cars_on_graph(sim: Simulation, n=5, seed=RANDOM_SEED):
    """
    Place cars *on graph nodes* when 7-col map coordinates are available.
    If coordinates are missing, we fall back to box seeding.
    """
    rng = random.Random(seed)
    node_coords = getattr(sim.map, "node_coordinates", {})
    if not node_coords:
        return seed_cars_box(sim, n=n, seed=seed)  # fallback

    sim.cars.clear()
    node_ids = list(node_coords.keys())
    for i in range(n):
        nid = rng.choice(node_ids)
        xy = node_coords[nid]
        cid = f"CAR{i+1:03d}"
        sim.cars[cid] = Car(cid, xy)
        sim.qtree_insert_car(sim.cars[cid])

def seed_cars_box(sim: Simulation, n=5, seed=RANDOM_SEED, bounds=(0.0, 0.0, 100.0, 100.0)):
    """
    Place cars uniformly at random inside a simple axis-aligned box (default 0..100).
    Used for legacy/no-map runs and as a safe fallback.
    """
    rng = random.Random(seed)
    minx, miny, maxx, maxy = bounds
    sim.cars.clear()
    for i in range(n):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        cid = f"CAR{i+1:03d}"
        sim.cars[cid] = Car(cid, (x, y))
        sim.qtree_insert_car(sim.cars[cid])

def seed_cars_auto(sim: Simulation, n=5, seed=RANDOM_SEED):
    """
    AUTO policy:
      - If 7-col map present => seed on graph nodes
      - Else => seed in default box
    """
    node_coords = getattr(sim.map, "node_coordinates", {})
    if node_coords:
        return seed_cars_on_graph(sim, n=n, seed=seed)
    return seed_cars_box(sim, n=n, seed=seed, bounds=(0.0, 0.0, 100.0, 100.0))

# CLI / Entrypoint

def parse_args():
    p = argparse.ArgumentParser(description="Ride-Sharing Simulation (final milestone)")
    p.add_argument("--map", default=None, help="Path to map.csv (7-col preferred; 3-col supported)")
    p.add_argument("--max-time", type=float, default=2000.0, help="Simulation horizon (time units)")
    p.add_argument("--mean-arrival", type=float, default=DEFAULT_MEAN_ARRIVAL, help="Mean inter-arrival time for riders")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")

    # Car seeding controls
    p.add_argument("--num-cars", type=int, default=5, help="How many cars to seed initially")
    p.add_argument("--seed-mode", choices=["auto", "graph", "box", "manual"], default="auto",
                   help="Where to place cars at start")
    p.add_argument("--manual-cars", type=str, default="",
                   help='Semicolon-separated (x,y) list for manual mode, e.g. "5,5;12,2;0,0"')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build the sim
    sim = Simulation(
        map_filename=args.map,
        max_time=args.max_time,
        mean_arrival=args.mean_arrival,
        seed=args.seed,
    )

    # Smart seeding policy
    if args.seed_mode == "manual":
        if not args.manual_cars.strip():
            raise SystemExit("--seed-mode manual requires --manual-cars like '5,5;12,2;0,0'")
        entries = [pair.strip() for pair in args.manual_cars.split(";") if pair.strip()]
        coords = []
        for e in entries:
            try:
                x_str, y_str = e.split(",")
                coords.append((float(x_str), float(y_str)))
            except Exception:
                raise SystemExit(f"Bad --manual-cars entry: {e} (expected 'x,y')")
        seed_cars_manual(sim, coords)
    elif args.seed_mode == "graph":
        seed_cars_on_graph(sim, n=args.num_cars, seed=args.seed)
    elif args.seed_mode == "box":
        seed_cars_box(sim, n=args.num_cars, seed=args.seed)
    else:
        seed_cars_auto(sim, n=args.num_cars, seed=args.seed)

    # Run the simulation + render PNG with KPIs
    sim.run()
    sim.render_summary_png("simulation_summary.png")
