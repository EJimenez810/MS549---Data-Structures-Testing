# simulation_demo.py
from simulation import Simulation
from car import Car
from rider import Rider

if __name__ == "__main__":
    # Map file and Graph loader (for testing will have it commented out)
    # sim = Simulation(map_filename="map.csv")
    sim = Simulation()

    # Seed cars
    sim.cars['CAR001'] = Car('CAR001', (5, 5))
    sim.cars['CAR002'] = Car('CAR002', (12, 2))
    sim.cars['CAR003'] = Car('CAR003', (0, 0))

    # Seed riders
    r1 = Rider('RIDER_A', (7,5), (10,8))
    r2 = Rider('RIDER_B', (2, 1), (3, 9))
    r3 = Rider('RIDER_C', (10, 2), (4, 2))
    sim.riders[r1.id] = r1
    sim.riders[r2.id] = r2
    sim.riders[r3.id] = r3

    # Schedule rider requests in seconds
    sim.add_event_at(10, 'RIDER_REQUEST', r1)
    sim.add_event_at(20, 'RIDER_REQUEST', r2)
    sim.add_event_at(25, 'RIDER_REQUEST', r3)

    # Run simulation
    sim.run()