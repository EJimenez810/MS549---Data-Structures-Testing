# pathfinding.py

import heapq

def find_shortest_path(graph, start_node, end_node):
    """
    Uses Dijkstra's algorithm to find the shortest path from start_node to end_node.
    Returns a tuple of (path as a list of nodes, total travel time).
    """
    distances = {node: float('inf') for node in graph.adjacency_list}
    predecessors = {}
    distances[start_node] = 0

    priority_queue = [(0, start_node)]  # (distance, node)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == end_node:
            # Found the shortest path to end_node
            break

        # Skip if a better path has already been found
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph.adjacency_list[current_node]:
            distance_through_current = current_distance + weight
            if distance_through_current < distances[neighbor]:
                distances[neighbor] = distance_through_current
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance_through_current, neighbor))

    # Reconstruct the path
    path = []
    current = end_node
    while current in predecessors:
        path.insert(0, current)
        current = predecessors[current]
    if path:
        path.insert(0, start_node)

    if distances[end_node] == float('inf'):
        return (None, float('inf'))  # No path found

    return (path, distances[end_node])
