# graph.py

import csv
import collections
from typing import Dict, List, Tuple, Iterable, Optional

class Graph:
    """
    City graph with both:
        Topology (adjacency list for weighted edges)
        Geometry (per-node (x, y) coordinates)

    Supports two input formats:
        Legacy 3-column CSV: start_node,end_node,weight        (directed)
        Unified 7-column CSV: start_node,start_x,start_y,end_node,end_x,end_y,weight
         (stored as undirected edges by default)
    """

    def __init__(self):
        # Adjacency list: node_id -> list[(neighbor_id, weight)]
        self.adjacency_list: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
        # Coordinates: node_id -> (x, y)
        self.node_coordinates: Dict[str, Tuple[float, float]] = {}

    # Core adjacency operations

    def add_directed_edge(self, start_node: str, end_node: str, weight: float) -> None:
        """Add a directed weighted edge."""
        self.adjacency_list[start_node].append((end_node, float(weight)))

    def add_undirected_edge(self, a: str, b: str, weight: float) -> None:
        """Add an undirected weighted edge (two directed edges)."""
        w = float(weight)
        self.adjacency_list[a].append((b, w))
        self.adjacency_list[b].append((a, w))

    def neighbors(self, node_id: str) -> Iterable[Tuple[str, float]]:
        return self.adjacency_list.get(node_id, ())

    def has_node(self, node_id: str) -> bool:
        return node_id in self.adjacency_list or node_id in self.node_coordinates

    # Legacy loader (3 columns)

    def load_from_file(self, filename: str) -> None:
        """
        Legacy loader for 3-column CSV: start_node,end_node,weight
        Adds *directed* edges. (If you want undirected, list both directions in the CSV.)
        """
        try:
            with open(filename, "r", newline="") as f:
                reader = csv.reader(_skip_blank_and_comment_lines(f))
                for row in reader:
                    if len(row) != 3:
                        # If the file is actually a 7-col file, suggest the new API.
                        raise ValueError(
                            f"Expected 3 columns, got {len(row)}. "
                            "Use load_map_data() for the 7-column unified format."
                        )
                    start, end, weight = [x.strip() for x in row]
                    self.add_directed_edge(start, end, float(weight))
            print("Map (3-column) loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"An error occurred in load_from_file: {e}")

    # Final milestone loader (7 columns)

    def load_map_data(self, filename: str) -> None:
        """
        Unified 7-column CSV with a header:
            start_node,start_x,start_y,end_node,end_x,end_y,weight
        Populates:
            node_coordinates for BOTH endpoints
            adjacency_list with UNDIRECTED edges (A<->B)
        """
        try:
            with open(filename, "r", newline="") as f:
                reader = csv.DictReader(f)
                # Validate required fieldnames (be forgiving on whitespace/case)
                expected = {"start_node", "start_x", "start_y", "end_node", "end_x", "end_y", "weight"}
                if reader.fieldnames is None or set(n.strip().lower() for n in reader.fieldnames) != expected:
                    # Attempt a soft check: allow any order but require same set
                    actual = set(n.strip().lower() for n in (reader.fieldnames or []))
                    if not expected.issubset(actual):
                        raise ValueError(
                            f"7-column header mismatch.\n"
                            f"Expected columns: {sorted(expected)}\n"
                            f"Found columns:    {sorted(actual)}"
                        )

                for row in reader:
                    if not row:
                        continue
                    # Read and cast fields
                    s = row["start_node"].strip()
                    t = row["end_node"].strip()
                    sx = float(row["start_x"]); sy = float(row["start_y"])
                    tx = float(row["end_x"]);   ty = float(row["end_y"])
                    w  = float(row["weight"])

                    # Store coordinates for both nodes
                    self.node_coordinates[s] = (sx, sy)
                    self.node_coordinates[t] = (tx, ty)

                    # Store undirected edge
                    self.add_undirected_edge(s, t, w)

            print("City map (7-column) loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"An error occurred in load_map_data: {e}")

    # Auto-detect loader (optional)
    
    def load(self, filename: str) -> None:
        """
        Auto-detects whether the CSV is 3 columns or 7 columns and calls
        the appropriate loader.
        """
        try:
            with open(filename, "r", newline="") as f:
                # Peek first non-blank, non-comment row
                for raw in f:
                    s = raw.strip()
                    if not s or s.startswith("#"):
                        continue
                    # If looks like header for 7-col, go 7-col path
                    lowered = [cell.strip().lower() for cell in s.split(",")]
                    if "start_node" in lowered and "end_node" in lowered:
                        self.load_map_data(filename)
                        return
                    # Else count columns
                    cols = len(lowered)
                    if cols == 7:
                        self.load_map_data(filename)
                        return
                    elif cols == 3:
                        self.load_from_file(filename)
                        return
                    else:
                        raise ValueError(f"Unrecognized CSV format for '{filename}' (found {cols} columns).")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"An error occurred in load(): {e}")

    # Convenience helpers
    
    def bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns (min_x, min_y, max_x, max_y) over all known node coordinates,
        or None if no coordinates loaded.
        """
        if not self.node_coordinates:
            return None
        xs = [xy[0] for xy in self.node_coordinates.values()]
        ys = [xy[1] for xy in self.node_coordinates.values()]
        return (min(xs), min(ys), max(xs), max(ys))

    def __str__(self) -> str:
        """
        Formatted view of the graph (topology + optional size of geometry).
        """
        lines = ["\n--- City Map (Adjacency List) ---"]
        for node, neighbors in self.adjacency_list.items():
            connections = ", ".join([f"({n}, {w})" for n, w in neighbors])
            lines.append(f"{node} -> [{connections}]")
        lines.append("--------------------------------")
        if self.node_coordinates:
            lines.append(f"Node coordinates: {len(self.node_coordinates)} nodes have (x, y)")
        return "\n".join(lines)


# Internal: skip blank lines and '#' comments in CSV (used by 3-col loader)
def _skip_blank_and_comment_lines(iterable):
    for raw in iterable:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        yield raw
