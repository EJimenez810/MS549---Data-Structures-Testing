# test_quadtree.py

import random
import time
from quadtree import Point, Rectangle, Quadtree

def brute_force_nearest(query_point, points):
    """Simple O(N) nearest neighbor for correctness verification."""
    best = None
    best_d2 = float('inf')
    for p in points:
        d2 = (p.x - query_point.x)**2 + (p.y - query_point.y)**2
        if d2 < best_d2:
            best_d2 = d2
            best = p
    return best, best_d2 ** 0.5

if __name__ == "__main__":
    random.seed(42)

    # 1) Build quadtree and populate with random points (drivers)
    boundary = Rectangle(0, 0, 1000, 1000)
    qt = Quadtree(boundary, capacity=4)

    num_points = 5000
    points = []
    for i in range(num_points):
        p = Point(random.uniform(0, 1000), random.uniform(0, 1000), data=f"Driver-{i}")
        points.append(p)
        qt.insert(p)

    # 2) Choose a query point (rider)
    rider = Point(512, 512, data="Rider")

    print(f"\n--- Nearest Neighbor Test ({num_points} points) ---")
    print(f"Query: {rider}\n")

    # 3) Quadtree search
    t0 = time.perf_counter()
    qt_best, qt_dist = qt.find_nearest(rider)
    t1 = time.perf_counter()

    # 4) Brute-force search
    t2 = time.perf_counter()
    bf_best, bf_dist = brute_force_nearest(rider, points)
    t3 = time.perf_counter()

    print(f"Quadtree:   best={qt_best},  dist={qt_dist:.4f},  time={(t1 - t0)*1000:.3f} ms")
    print(f"BruteForce: best={bf_best},  dist={bf_dist:.4f},  time={(t3 - t2)*1000:.3f} ms")

    # 5) Verify correctness (allow tiny floating-point tolerance)
    same_point = (qt_best is bf_best) or (abs(qt_dist - bf_dist) < 1e-9)
    print("\nCorrectness:", "PASS" if same_point else "FAIL")