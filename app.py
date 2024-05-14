from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from simanneal import Annealer
from math import radians, sin, cos, sqrt, atan2
import urllib.parse
from datetime import datetime

app = Flask(__name__)

# Load CSV data
data = pd.read_csv('locations_updated.txt')

# Extract location names, latitudes, and longitudes
location_names = data['Address'].tolist()
latitudes = data['Latitude'].tolist()
longitudes = data['Longitude'].tolist()
customer_names = data['Customer_Name'].tolist()

# Get the number of locations
n_locations = len(location_names)

# Define a function to calculate the Haversine distance between two points
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Calculate distance matrix
distance_matrix = np.zeros((n_locations, n_locations))
for i in range(n_locations):
    for j in range(n_locations):
        if i != j:
            distance_matrix[i, j] = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

# Define a custom simulated annealing class for TSP
class TSPAnnealer(Annealer):
    def __init__(self, state):
        super(TSPAnnealer, self).__init__(state)
    
    def move(self):
        """Swap two cities."""
        idx1, idx2 = np.random.choice(len(self.state), 2, replace=False)
        self.state[idx1], self.state[idx2] = self.state[idx2], self.state[idx1]

    def energy(self):
        """Calculate the total distance of the route."""
        total_distance = 0
        for i in range(len(self.state) - 1):
            total_distance += distance_matrix[self.state[i], self.state[i + 1]]
        total_distance += distance_matrix[self.state[-1], self.state[0]]
        return total_distance

# Define the initial state
initial_state = list(range(n_locations))

# Create an instance of the TSPAnnealer with the initial state
annealer = TSPAnnealer(initial_state)

# Run the simulated annealing algorithm
annealer.set_schedule(annealer.auto(minutes=5))

# Get the optimized route and distance
optimized_route, optimized_distance = annealer.anneal()

# Generate the Google Maps URL
def generate_google_maps_url(route, location_names, latitudes, longitudes):
    base_url = "https://www.google.com/maps/dir/?api=1"
    travel_mode = "driving"
    waypoints = []

    # Start point
    start_point = f"{latitudes[route[0]]},{longitudes[route[0]]}"

    # Collect waypoints
    for idx in route:
        waypoint = f"{latitudes[idx]},{longitudes[idx]}"
        waypoints.append(waypoint)

    # Join the waypoints using the pipe ('|') delimiter
    waypoints_str = '|'.join(waypoints)

    # Create the URL
    url = f"{base_url}&origin={start_point}&destination={start_point}&waypoints={urllib.parse.quote(waypoints_str)}&travelmode={travel_mode}"

    return url

google_maps_url = generate_google_maps_url(optimized_route, location_names, latitudes, longitudes)

# Prepare the response
response = {
    'optimized_route': optimized_route,
    'optimized_distance': optimized_distance,
    'google_maps_url': google_maps_url,
    'location_names': location_names,
    'customer_names': customer_names,  # Include customer names
    'current_date': datetime.now().strftime("%Y-%m-%d")
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return render_template('results.html', response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
