# MS549---Data-Structures-Testing
Ride-Sharing Simulator Application

## Project Title
Efficient Ride-Sharing Simulator – Object-Oriented Design (Milestone 1)

## Purpose / Design
This project is the first milestone of a multi-phase simulation designed to mimic the behavior of a ride-sharing system using object-oriented programming (OOP). This initial version defines three foundational classes: `Car`, `Rider`, and `Simulation`, which serve as the building blocks for future algorithmic features like ride matching and route optimization.

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

In this milestone, we introduced a `Graph` class to represent the city map as an adjacency list, allowing for flexible and dynamic map loading from an external `.csv` file. The `Simulation` class was updated to accept a `map_filename` parameter and now stores the loaded graph as `self.map`.

## Milestone 3: Dijkstra's pathfinding test

In this milestone, we implemented Dijkstra's Shortest Path Algorithm to enable cars to intelligently navigate the city map. A new `pathfinding.py` module contains the core logic for efficiently finding the shortest path between two nodes using a priority queue (min-heap). The `Car` class was enhanced with a `calculate_route(destination, graph)` method, allowing each car to compute its optimal route and total travel time to a given destination based on the loaded city map. Test scripts demonstrate the pathfinding functionality by calculating routes for cars and verifying the results through printed path sequences and travel times. This milestone establishes the foundational pathfinding capability necessary for dynamic ride assignments in future phases of the simulator.

### 🔧 How to Run

1. Clone or download this repository.
2. Ensure `map.csv` exists in your project directory.
3. Open a terminal or command prompt.
4. Run the updated test script with:

```bash
python3 test_simulation.py
python test_graph_simulation.py
python3 test_dijkstra.py
