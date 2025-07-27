# simulation.py

from graph import Graph

# Defines the Simulation class  
class Simulation:
    """
    Manages the simulation including cars, riders, and the city map.
    """

    def __init__(self, map_filename):
        # Dictionary to store Cars by ID      
        self.cars = {}
        
        # Dictionary to store Riders by ID
        self.riders = {}

        # Load map from CSV file
        self.map = Graph()
        self.map.load_from_file(map_filename)

    def __str__(self):
        # Print a summary of how many cars and riders are in the simulation
       return (f"Simulation with {len(self.cars)} cars and "
                f"{len(self.riders)} riders.\n{self.map}")
