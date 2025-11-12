"""
Generate synthetic test data for when OSM/Overpass API is unavailable.
This allows testing the pipeline without external API calls.
"""
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from config import WROCLAW_BBOX, CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_network():
    """Create a synthetic pedestrian network."""
    logger.info("Creating synthetic pedestrian network...")
    
    G = nx.MultiDiGraph()
    
    # Create a grid network
    bbox = WROCLAW_BBOX
    lats = np.linspace(bbox['south'], bbox['north'], 20)
    lons = np.linspace(bbox['west'], bbox['east'], 20)
    
    node_id = 0
    node_map = {}
    
    # Create nodes
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            G.add_node(node_id, y=lat, x=lon, street_count=4)
            node_map[(i, j)] = node_id
            node_id += 1
    
    # Create edges (grid connections)
    for i in range(len(lats)):
        for j in range(len(lons)):
            current = node_map[(i, j)]
            # Connect to right neighbor
            if j < len(lons) - 1:
                neighbor = node_map[(i, j+1)]
                G.add_edge(current, neighbor, length=100, highway='residential')
                G.add_edge(neighbor, current, length=100, highway='residential')
            # Connect to upper neighbor
            if i < len(lats) - 1:
                neighbor = node_map[(i+1, j)]
                G.add_edge(current, neighbor, length=100, highway='residential')
                G.add_edge(neighbor, current, length=100, highway='residential')
    
    logger.info(f"Created synthetic network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G

def create_synthetic_sidewalks():
    """Create synthetic sidewalk data."""
    logger.info("Creating synthetic sidewalks...")
    
    bbox = WROCLAW_BBOX
    geometries = []
    
    # Create some synthetic sidewalks
    for _ in range(100):
        lat = np.random.uniform(bbox['south'], bbox['north'])
        lon = np.random.uniform(bbox['west'], bbox['east'])
        lat2 = lat + np.random.uniform(-0.001, 0.001)
        lon2 = lon + np.random.uniform(-0.001, 0.001)
        geometries.append(LineString([(lon, lat), (lon2, lat2)]))
    
    gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')
    logger.info(f"Created {len(gdf)} synthetic sidewalks")
    return gdf

def create_synthetic_crosswalks():
    """Create synthetic crosswalk data."""
    logger.info("Creating synthetic crosswalks...")
    
    bbox = WROCLAW_BBOX
    geometries = []
    
    # Create some synthetic crosswalks
    for _ in range(50):
        lat = np.random.uniform(bbox['south'], bbox['north'])
        lon = np.random.uniform(bbox['west'], bbox['east'])
        geometries.append(Point(lon, lat))
    
    gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')
    gdf['crossing'] = 'marked'
    logger.info(f"Created {len(gdf)} synthetic crosswalks")
    return gdf

def create_synthetic_amenities():
    """Create synthetic amenity data."""
    logger.info("Creating synthetic amenities...")
    
    bbox = WROCLAW_BBOX
    amenity_types = ['supermarket', 'school', 'park', 'pharmacy', 'restaurant', 'cafe']
    
    data = []
    for _ in range(200):
        lat = np.random.uniform(bbox['south'], bbox['north'])
        lon = np.random.uniform(bbox['west'], bbox['east'])
        amenity = np.random.choice(amenity_types)
        data.append({
            'geometry': Point(lon, lat),
            'amenity': amenity,
            'name': f'Test {amenity} {_}'
        })
    
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    logger.info(f"Created {len(gdf)} synthetic amenities")
    return gdf

def create_synthetic_neighborhoods():
    """Create synthetic neighborhood boundaries."""
    logger.info("Creating synthetic neighborhoods...")
    
    bbox = WROCLAW_BBOX
    neighborhoods = []
    
    # Create a 3x3 grid of neighborhoods
    lat_step = (bbox['north'] - bbox['south']) / 3
    lon_step = (bbox['east'] - bbox['west']) / 3
    
    idx = 0
    for i in range(3):
        for j in range(3):
            south = bbox['south'] + i * lat_step
            north = south + lat_step
            west = bbox['west'] + j * lon_step
            east = west + lon_step
            
            poly = Polygon([
                (west, south),
                (east, south),
                (east, north),
                (west, north),
                (west, south)
            ])
            
            neighborhoods.append({
                'geometry': poly,
                'name': f'Neighborhood_{idx}',
                'admin_level': 10
            })
            idx += 1
    
    gdf = gpd.GeoDataFrame(neighborhoods, crs='EPSG:4326')
    logger.info(f"Created {len(gdf)} synthetic neighborhoods")
    return gdf

def create_synthetic_transit_stops():
    """Create synthetic transit stop data."""
    logger.info("Creating synthetic transit stops...")
    
    bbox = WROCLAW_BBOX
    data = []
    
    for _ in range(30):
        lat = np.random.uniform(bbox['south'], bbox['north'])
        lon = np.random.uniform(bbox['west'], bbox['east'])
        data.append({
            'geometry': Point(lon, lat),
            'stop_name': f'Stop {_}',
            'stop_id': f'STOP_{_}'
        })
    
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    logger.info(f"Created {len(gdf)} synthetic transit stops")
    return gdf

def save_all_synthetic_data():
    """Generate and save all synthetic data to cache."""
    import osmnx as ox
    
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Network
    G = create_synthetic_network()
    ox.save_graphml(G, Path(CACHE_DIR) / 'wroclaw_walk_network.graphml')
    
    # Sidewalks
    sidewalks = create_synthetic_sidewalks()
    sidewalks.to_file(Path(CACHE_DIR) / 'wroclaw_sidewalks.gpkg', driver='GPKG')
    
    # Crosswalks
    crosswalks = create_synthetic_crosswalks()
    crosswalks.to_file(Path(CACHE_DIR) / 'wroclaw_crosswalks.gpkg', driver='GPKG')
    
    # Amenities
    amenities = create_synthetic_amenities()
    amenities.to_file(Path(CACHE_DIR) / 'wroclaw_amenities.gpkg', driver='GPKG')
    
    # Neighborhoods
    neighborhoods = create_synthetic_neighborhoods()
    neighborhoods.to_file(Path(CACHE_DIR) / 'wroclaw_neighborhoods.gpkg', driver='GPKG')
    
    # Transit stops
    transit_stops = create_synthetic_transit_stops()
    transit_stops.to_file(Path(CACHE_DIR) / 'wroclaw_transit_stops.gpkg', driver='GPKG')
    
    logger.info("All synthetic data saved to cache!")

if __name__ == "__main__":
    save_all_synthetic_data()
