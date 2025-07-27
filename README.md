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
- No external libraries are required for this milestone.

## Files Included
- `car.py`
- `rider.py`
- `simulation.py`
- `test_simulation.py`
- `map.csv`
- `graph.py`
- `test_graph_simulation.py`

## Milestone 2: Graph Integration – City Map Support

In this milestone, we introduced a `Graph` class to represent the city map as an adjacency list, allowing for flexible and dynamic map loading from an external `.csv` file. The `Simulation` class was updated to accept a `map_filename` parameter and now stores the loaded graph as `self.map`.

### 🔧 How to Run

1. Ensure `map.csv` exists in your project directory.
2. Run the updated test script with:

```bash
python3 test_simulation.py
python test_graph_simulation.py
