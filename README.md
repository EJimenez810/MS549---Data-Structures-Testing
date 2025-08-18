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
