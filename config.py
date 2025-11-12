"""
Configuration file for Wrocław Walkability Analyzer.
Contains all constants, parameters, and settings for the project.
"""

# Geographic Boundaries for Wrocław
# Using smaller area for faster testing (central Rynek area)
WROCLAW_BBOX = {
    'north': 51.115,
    'south': 51.105,
    'east': 17.045,
    'west': 17.025
}

# Full city bounding box (use for production):
# WROCLAW_BBOX = {
#     'north': 51.15,
#     'south': 51.05,
#     'east': 17.15,
#     'west': 16.95
# }

# Center point of Wrocław for reference
WROCLAW_CENTER = (51.1079, 17.0385)

# Data Collection Settings
OSM_NETWORK_TYPE = 'walk'  # Pedestrian network
CACHE_DIR = 'data/cache'
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
GTFS_DATA_DIR = 'data/gtfs'

# Feature Engineering Parameters
AMENITY_TYPES = [
    'supermarket',
    'school',
    'kindergarten',
    'park',
    'pharmacy',
    'cafe',
    'restaurant',
    'library'
]

AMENITY_TAGS = {
    'amenity': AMENITY_TYPES,
    'leisure': ['park', 'playground']
}

# OSM Tags for Infrastructure
SIDEWALK_TAGS = {
    'highway': ['footway', 'pedestrian', 'path']
}

CROSSWALK_TAGS = {
    'highway': 'crossing'
}

# ML Model Settings
MODEL_DIR = 'models'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Walkability Scoring Weights (for fallback/baseline)
SCORE_WEIGHTS = {
    'infrastructure': 0.40,  # Sidewalks, crosswalks
    'amenities': 0.30,       # Proximity to services
    'transit': 0.20,         # Public transit access
    'safety': 0.10          # Other factors
}

# Distance thresholds (in meters)
MAX_AMENITY_DISTANCE = 1000  # 1 km
MAX_TRANSIT_DISTANCE = 500   # 500 m
WALKING_SPEED_MPS = 1.4      # Average walking speed (m/s)

# Neighborhoods to analyze (sample list)
TARGET_NEIGHBORHOODS = [
    'Stare Miasto',
    'Nadodrze',
    'Krzyki',
    'Przedmieście Oławskie',
    'Fabryczna',
    'Psie Pole',
    'Śródmieście',
    'Karłowice',
    'Borek',
    'Gaj'
]

# Output Settings
OUTPUT_DIR = 'outputs'
FEATURE_FILE = f'{PROCESSED_DATA_DIR}/neighborhood_features.csv'
SCORE_FILE = f'{PROCESSED_DATA_DIR}/walkability_scores.csv'
MODEL_FILE = f'{MODEL_DIR}/walkability_model.pkl'

# Visualization Settings
MAP_TILE = 'OpenStreetMap'
SCORE_COLORMAP = 'RdYlGn'  # Red-Yellow-Green for scores

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
