# test_dijkstra.py

from graph import Graph
from car import Car

if __name__ == "__main__":
    # Load the city map
    city_map = Graph()
    city_map.load_from_file('map.csv')

    # Display the graph to verify it loaded correctly
    print(city_map)

    # Create a car at a starting point 'A'
    car1 = Car('CAR001', 'A')

    # Calculate route from 'A' to 'D' using Dijkstra's algorithm
    car1.calculate_route('D', city_map)

    # Print the calculated route and total travel time
    print(f"Calculated Route for {car1.id}: {car1.route}")
    print(f"Total Travel Time: {car1.route_time}")