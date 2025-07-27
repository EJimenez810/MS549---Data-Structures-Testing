# graph.py

import csv

class Graph:
    """
    A class to represent a directed, weighted graph using an adjacency list.
    """

    def __init__(self):
        # Each node maps to a list of (neighbor, weight) tuples
        self.adjacency_list = {}

    def add_edge(self, start_node, end_node, weight):
        """
        Adds a directed edge from start_node to end_node with the given weight.
        """
        if start_node not in self.adjacency_list:
            self.adjacency_list[start_node] = []
        self.adjacency_list[start_node].append((end_node, float(weight)))

    def load_from_file(self, filename):
        """
        Loads the graph structure from a CSV file: start_node,end_node,weight
        """
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 3: # Ensure the row has 3 elements
                        start, end, weight = row
                        self.add_edge(start.strip(), end.strip(), weight.strip())
            print("Map loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __str__(self):
        """
        Nicely formats the graph for display.
        """
        output = ["\n--- City Map (Adjacency List) ---"]
        for node, neighbors in self.adjacency_list.items():
            connections = ", ".join([f"({n}, {w})" for n, w in neighbors])
            output.append(f"{node} -> [{connections}]")
        output.append("--------------------------------")
        return "\n".join(output)
