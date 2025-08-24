# MS549---Data-Structures-Testing
Ride-Sharing Simulator Application

## Project Title
Efficient Ride-Sharing Simulator – Object-Oriented Design (Milestone 1)

## Purpose / Design
This project is the first milestone of a multi-phase simulation designed to mimic the behavior of a ride-sharing system using object-oriented programming (OOP). This initial version defines three foundational classes: Car, Rider, and Simulation, which serve as the building blocks for future algorithmic features like ride matching and route optimization.

## How to Run

1. Clone or download this repository.
2. Open a terminal or command prompt.
3. Navigate to the project directory.
4. Run the test script with:

```bash
python3 test_simulation.py
```

This script creates and prints instances of `Car`, `Rider`, and `Simulation` to verify the basic structure and functionality.

## Dependencies
- Python 3.x
- Uses Python's built-in `heapq` and `csv` modules
- No external libraries are required.

## Files Included
- `car.py`
- `rider.py`
- `simulation.py`
- `test_simulation.py`
- `map.csv`
- `graph.py`
- `test_graph_simulation.py`
- `pathfinding.py`
- `test_dijkstra.py`

## Map Data Format (map.csv)
The city map is stored in a CSV file with the following format:

start_node,end_node,travel_time
Example:

A,B,5
B,A,5
A,C,3
C,A,3
B,D,4
D,B,4
C,D,1
D,C,1

For bidirectional roads, both directions must be specified.

## Milestone 2: Graph Integration – City Map Support

Introduced a `Graph` class to represent the city map as an adjacency list, allowing for flexible and dynamic map loading from an external `.csv` file. The `Simulation` class was updated to accept a `map_filename` parameter and now stores the loaded graph as `self.map`.

## Milestone 3: Dijkstra's pathfinding test

In this milestone, we implemented Dijkstra's Shortest Path Algorithm to enable cars to navigate the city map intelligently. A new `pathfinding.py` module contains the core logic for efficiently finding the shortest path between two nodes using a priority queue (min-heap). The `Car` class was enhanced with a `calculate_route(destination, graph)` method, allowing each car to compute its optimal route and total travel time to a given destination based on the loaded city map. Test scripts demonstrate the pathfinding functionality by calculating routes for cars and verifying the results through printed path sequences and travel times. This milestone establishes the foundational pathfinding capability necessary for dynamic ride assignments in future phases of the simulator.

### 🔧 How to Run

1. Clone or download this repository.
2. Ensure `map.csv` exists in your project directory.
3. Open a terminal or command prompt.
4. Run the updated test script with:

```bash
python3 test_simulation.py
python test_graph_simulation.py
python3 test_dijkstra.py
```

## Milestone 5: Quadtree – Efficient Nearest-Neighbor Search

This milestone introduces a spatial index using a **Quadtree** to find the nearest available car to a rider in approximately O(log N) time (vs. O(N) brute-force). The tree partitions the 2D map into recursively smaller rectangles and prunes entire regions that cannot contain a closer point.

Files Added
- `quadtree.py` – Manual classes (`Point`, `Rectangle`, `QuadtreeNode`, `Quadtree`), no dataclasses used, might be changed to dataclasses in future updates.
- `test_quadtree.py` – Standalone test that compares Quadtree results to a brute-force search for correctness and prints timings.

## How to Run
```bash
python3 test_quadtree.py
```

## Milestone 6: Simulation Engine Prototype
In this milestone, we introduced an event-driven simulation engine to model ride-sharing dynamics over time. Instead of running all logic instantly, events (like rider requests, car arrivals, and drop-offs) are scheduled and executed in chronological order using a priority queue (heapq).

## Key Features:
1. `add_event(delay, event_type, data)` schedules events relative to the current simulation time.
2. `run()` processes events in order, advancing simulation time automatically.
3. `handle_rider_request(rider_id)` assigns available cars to riders (first-come-first-serve).
4. `handle_arrival(data)` manages pickup and drop-off logic, updating car status back to “available” after drop-off.
5. Uses a placeholder Manhattan distance travel time function until full routing is integrated.

## Files Modified:
- `simulation.py` – now contains the event loop and event handlers.
- `car.py` – updated with assigned_rider attribute to track active assignments.
- `rider.py` – slight modification from prior milestone.

## How to Run
```bash
python3 simulation_demo.py
```

## Final Project Overview:
This project implements a discrete-event ride-sharing simulator in Python, developed incrementally over multiple milestones. The final version integrates cars, riders, a city graph with coordinates, Dijkstra’s shortest-path algorithm, a spatial index (Quadtree), and dynamic event-driven request generation. The simulator models how a ride-sharing service (like Uber/Lyft) operates under load, collects performance data (KPIs), and outputs a visual analytical summary (`simulation_summary.png`).

## Final Features:
1. Dynamic Event-Driven Simulation
    - Rider requests generated with random inter-arrival times (Poisson process).
    - Events managed by a priority queue (`heapq`).
2. Graph + Dijkstra Pathfinding
    - Supports 7-column unified CSV maps with both node IDs and (x,y) coordinates (`city_map.csv`).
    - Travel times computed using shortest-path routing, not just Manhattan distance.
3. Car Matching
    - Efficient Quadtree nearest-neighbor search (O(log N)) to find candidate cars.
    - Falls back to brute-force search if Quadtree payloads aren’t supported.
    - Chooses the best car using Dijkstra’s true driving time.
4. Performance Metrics (KPIs)
    - Average rider wait time.
    - Average trip duration.
    - Driver utilization (% of time cars are busy).
5. Visualization Output
    - Final car positions overlaid on the city map (`simulation_summary.png`).
    - Performance metrics displayed as text (`simulation_summary.png` and terminal text).
    - Histogram of rider wait times included for deeper insight (`simulation_summary.png`).
  
## Files Modified:
- `car.py`
- `rider.py`
- `graph.py`
- `map.csv` - Now renamed `legacy_map.csv`
- `city_map.csv` - New file
- `simulation.py`
- `simulation_summary.png` - File created when running simulation

## Full Project Structure:
- `car.py` – Defines the Car class (status, assigned rider, route).
- `rider.py` – Defines the Rider class (start, destination, status).
- `graph.py` – City map with adjacency list + node coordinates; loads both 3-col legacy and 7-col unified CSVs.
- `pathfinding.py` – Dijkstra’s algorithm for shortest-path routing.
- `quadtree.py` – Quadtree spatial index for efficient car lookup.
- `simulation.py` – Final event-driven engine (dynamic requests, car assignment, KPIs, visualization).
- `city_map.csv` – Final 7-column city map (coordinates + edges).
- `legacy_map.csv` – Legacy 3-column format, kept for backwards compatibility.

## How to Run
Basic run with default map and settings
```bash
python3 simulation.py --map city_map.csv --num-cars 5 --max-time 600
```

## Command-Line Options

1. `--map` → Path to `city_map.csv` (7-col preferred).
2. `--max-time` → Simulation horizon (time units).
3. `--mean-arrival` → Mean inter-arrival time for riders (default = 30).
4. `--num-cars` → Number of cars to seed initially.
5. `--seed-mode` → How to place cars:
    1. `auto` → On graph nodes if 7-col map available, else in default box.
    2. `graph` → Exactly on random graph nodes.
    3. `box` → Uniform random in `[0,100]^2`.
    4. `manual` → User-specified positions via `--manual-cars`.
6. `--manual-cars` → Positions for manual mode, e.g. `"5,5;12,2;0,0"`.
7. `--seed` → Random seed for reproducibility.

## Example Runs
Run with graph-based seeding
```bash
python3 simulation.py --map city_map.csv --num-cars 10 --seed-mode graph
```
Run with manual cars
```bash
python3 simulation.py --map city_map.csv --seed-mode manual --manual-cars "5,5;12,2;0,0"
```

## Output:
After the simulation ends, a single PNG file is generated:
`simulation_summary.png`
  - Left panel: City map + final car positions.
  - Right panel: KPI summary and histogram of rider wait times.

## Dependencies
1. Python 3.x
    - Standard library:
        - `heapq`
        - `csv`
        - `math`
        - `argparse`
        - `random`
    - External:
        - `matplotlib` - for visualization
2. Install Dependencies:
    ```bash
    pip install matplotlib
    ```
