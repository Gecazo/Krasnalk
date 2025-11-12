"""
Data Gathering Pipeline for Wrocław Walkability Analyzer
========================================================

This module fetches and processes geospatial data from OpenStreetMap (OSM) 
and GTFS transit data to create features for walkability analysis.

Main Steps:
1. Fetch pedestrian network and infrastructure from OSM
2. Extract amenities and neighborhood boundaries
3. Parse GTFS data for public transit stops
4. Calculate walkability features per neighborhood
5. Save processed data for ML modeling

"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Fix Windows encoding issues with Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm

# Import configuration
from config import (
    WROCLAW_BBOX, WROCLAW_CENTER, OSM_NETWORK_TYPE,
    CACHE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, GTFS_DATA_DIR,
    AMENITY_TAGS, SIDEWALK_TAGS, CROSSWALK_TAGS,
    MAX_AMENITY_DISTANCE, MAX_TRANSIT_DISTANCE, TARGET_NEIGHBORHOODS,
    FEATURE_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure OSMnx
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.cache_folder = CACHE_DIR


class WroclawDataGatherer:
    """Main class for gathering and processing Wrocław walkability data."""
    
    def __init__(self):
        """Initialize the data gatherer with bbox and paths."""
        self.bbox = WROCLAW_BBOX
        self.center = WROCLAW_CENTER
        
        # Ensure directories exist
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(GTFS_DATA_DIR).mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.pedestrian_graph = None
        self.sidewalks_gdf = None
        self.crosswalks_gdf = None
        self.amenities_gdf = None
        self.neighborhoods_gdf = None
        self.transit_stops_gdf = None
        
        logger.info("WroclawDataGatherer initialized")
    
    def fetch_pedestrian_network(self) -> nx.MultiDiGraph:
        """
        Fetch pedestrian/walking network from OSM.
        
        Returns:
            NetworkX MultiDiGraph representing the walkable network
        """
        logger.info("Fetching pedestrian network from OSM...")
        
        try:
            # Try to load from cache
            cache_path = Path(CACHE_DIR) / 'wroclaw_walk_network.graphml'
            
            if cache_path.exists():
                logger.info("Loading pedestrian network from cache...")
                self.pedestrian_graph = ox.load_graphml(cache_path)
            else:
                # Fetch from OSM
                # OSMnx 2.x expects bbox as tuple (north, south, east, west)
                bbox_tuple = (
                    self.bbox['north'],
                    self.bbox['south'],
                    self.bbox['east'],
                    self.bbox['west']
                )
                self.pedestrian_graph = ox.graph_from_bbox(
                    bbox_tuple,
                    network_type=OSM_NETWORK_TYPE,
                    simplify=True
                )
                
                # Save to cache
                ox.save_graphml(self.pedestrian_graph, cache_path)
                logger.info(f"Saved pedestrian network to cache: {cache_path}")
            
            # Get network statistics
            n_nodes = len(self.pedestrian_graph.nodes())
            n_edges = len(self.pedestrian_graph.edges())
            logger.info(f"Pedestrian network: {n_nodes} nodes, {n_edges} edges")
            
            return self.pedestrian_graph
            
        except Exception as e:
            logger.error(f"Error fetching pedestrian network: {e}")
            raise
    
    def fetch_sidewalks(self) -> gpd.GeoDataFrame:
        """
        Fetch sidewalk and footway geometries from OSM.
        
        Returns:
            GeoDataFrame containing sidewalk geometries
        """
        logger.info("Fetching sidewalks from OSM...")
        
        try:
            # Try GPKG cache first (from synthetic data), then GeoJSON
            cache_path_gpkg = Path(CACHE_DIR) / 'wroclaw_sidewalks.gpkg'
            cache_path_geojson = Path(CACHE_DIR) / 'wroclaw_sidewalks.geojson'
            
            if cache_path_gpkg.exists():
                logger.info("Loading sidewalks from GPKG cache...")
                self.sidewalks_gdf = gpd.read_file(cache_path_gpkg)
            elif cache_path_geojson.exists():
                logger.info("Loading sidewalks from GeoJSON cache...")
                self.sidewalks_gdf = gpd.read_file(cache_path_geojson)
            else:
                # Fetch from OSM using features_from_bbox (OSMnx 2.x)
                bbox_tuple = (
                    self.bbox['north'],
                    self.bbox['south'],
                    self.bbox['east'],
                    self.bbox['west']
                )
                gdf = ox.features_from_bbox(
                    bbox_tuple,
                    tags=SIDEWALK_TAGS
                )
                
                # Filter to LineStrings only (sidewalks are lines)
                self.sidewalks_gdf = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
                
                # Save to cache
                self.sidewalks_gdf.to_file(cache_path_geojson, driver='GeoJSON')
                logger.info(f"Saved sidewalks to cache: {cache_path_geojson}")
            
            logger.info(f"Found {len(self.sidewalks_gdf)} sidewalk segments")
            return self.sidewalks_gdf
            
        except Exception as e:
            logger.warning(f"Error fetching sidewalks (may be sparse data): {e}")
            # Return empty GeoDataFrame if no data
            self.sidewalks_gdf = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
            return self.sidewalks_gdf
    
    def fetch_crosswalks(self) -> gpd.GeoDataFrame:
        """
        Fetch crosswalk/crossing points from OSM.
        
        Returns:
            GeoDataFrame containing crosswalk points
        """
        logger.info("Fetching crosswalks from OSM...")
        
        try:
            cache_path_gpkg = Path(CACHE_DIR) / 'wroclaw_crosswalks.gpkg'
            cache_path = Path(CACHE_DIR) / 'wroclaw_crosswalks.geojson'
            
            if cache_path_gpkg.exists():
                logger.info("Loading crosswalks from GPKG cache...")
                self.crosswalks_gdf = gpd.read_file(cache_path_gpkg)
            elif cache_path.exists():
                logger.info("Loading crosswalks from GeoJSON cache...")
                self.crosswalks_gdf = gpd.read_file(cache_path)
            else:
                # Fetch from OSM using features_from_bbox (OSMnx 2.x)
                bbox_tuple = (
                    self.bbox['north'],
                    self.bbox['south'],
                    self.bbox['east'],
                    self.bbox['west']
                )
                gdf = ox.features_from_bbox(
                    bbox_tuple,
                    tags=CROSSWALK_TAGS
                )
                
                # Filter to Points only
                self.crosswalks_gdf = gdf[gdf.geometry.type == 'Point'].copy()
                
                # Save to cache
                self.crosswalks_gdf.to_file(cache_path, driver='GeoJSON')
                logger.info(f"Saved crosswalks to cache: {cache_path}")
            
            logger.info(f"Found {len(self.crosswalks_gdf)} crosswalks")
            return self.crosswalks_gdf
            
        except Exception as e:
            logger.warning(f"Error fetching crosswalks: {e}")
            self.crosswalks_gdf = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
            return self.crosswalks_gdf
    
    def fetch_amenities(self, max_pois: int = 500) -> gpd.GeoDataFrame:
        """
        Fetch amenities (shops, schools, parks, etc.) from OSM.
        
        Args:
            max_pois: Maximum number of POIs to fetch (for performance)
        
        Returns:
            GeoDataFrame containing amenity points
        """
        logger.info("Fetching amenities from OSM...")
        
        try:
            cache_path_gpkg = Path(CACHE_DIR) / 'wroclaw_amenities.gpkg'
            cache_path = Path(CACHE_DIR) / 'wroclaw_amenities.geojson'
            
            if cache_path_gpkg.exists():
                logger.info("Loading amenities from GPKG cache...")
                self.amenities_gdf = gpd.read_file(cache_path_gpkg)
            elif cache_path.exists():
                logger.info("Loading amenities from GeoJSON cache...")
                self.amenities_gdf = gpd.read_file(cache_path)
            else:
                # Fetch from OSM using features_from_bbox (OSMnx 2.x)
                bbox_tuple = (
                    self.bbox['north'],
                    self.bbox['south'],
                    self.bbox['east'],
                    self.bbox['west']
                )
                gdf = ox.features_from_bbox(
                    bbox_tuple,
                    tags=AMENITY_TAGS
                )
                
                # Convert to points (use centroids for polygons)
                gdf_points = gdf.copy()
                gdf_points['geometry'] = gdf_points.geometry.centroid
                
                # Sample if too many
                if len(gdf_points) > max_pois:
                    logger.info(f"Sampling {max_pois} from {len(gdf_points)} amenities")
                    self.amenities_gdf = gdf_points.sample(n=max_pois, random_state=42)
                else:
                    self.amenities_gdf = gdf_points
                
                # Save to cache
                self.amenities_gdf.to_file(cache_path, driver='GeoJSON')
                logger.info(f"Saved amenities to cache: {cache_path}")
            
            logger.info(f"Found {len(self.amenities_gdf)} amenities")
            return self.amenities_gdf
            
        except Exception as e:
            logger.error(f"Error fetching amenities: {e}")
            self.amenities_gdf = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
            return self.amenities_gdf
    
    def fetch_neighborhoods(self) -> gpd.GeoDataFrame:
        """
        Fetch neighborhood boundaries from OSM or create synthetic ones.
        
        Returns:
            GeoDataFrame containing neighborhood polygons
        """
        logger.info("Fetching neighborhood boundaries...")
        
        try:
            cache_path_gpkg = Path(CACHE_DIR) / 'wroclaw_neighborhoods.gpkg'
            cache_path = Path(CACHE_DIR) / 'wroclaw_neighborhoods.geojson'
            
            if cache_path_gpkg.exists():
                logger.info("Loading neighborhoods from GPKG cache...")
                self.neighborhoods_gdf = gpd.read_file(cache_path_gpkg)
            elif cache_path.exists():
                logger.info("Loading neighborhoods from GeoJSON cache...")
                self.neighborhoods_gdf = gpd.read_file(cache_path)
            else:
                # Try to fetch admin boundaries
                try:
                    bbox_tuple = (
                        self.bbox['north'],
                        self.bbox['south'],
                        self.bbox['east'],
                        self.bbox['west']
                    )
                    gdf = ox.features_from_bbox(
                        bbox_tuple,
                        tags={'admin_level': '9', 'boundary': 'administrative'}
                    )
                    
                    # Filter to polygons
                    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
                    
                    # Filter by name if available
                    if 'name' in gdf.columns:
                        gdf = gdf[gdf['name'].notna()].copy()
                    
                    self.neighborhoods_gdf = gdf
                    
                except Exception as e:
                    logger.warning(f"Could not fetch admin boundaries: {e}")
                    # Create synthetic grid neighborhoods
                    self.neighborhoods_gdf = self._create_synthetic_neighborhoods()
                
                # Save to cache
                self.neighborhoods_gdf.to_file(cache_path, driver='GeoJSON')
                logger.info(f"Saved neighborhoods to cache: {cache_path}")
            
            logger.info(f"Found {len(self.neighborhoods_gdf)} neighborhoods")
            return self.neighborhoods_gdf
            
        except Exception as e:
            logger.error(f"Error fetching neighborhoods: {e}")
            # Fallback to synthetic
            self.neighborhoods_gdf = self._create_synthetic_neighborhoods()
            return self.neighborhoods_gdf
    
    def _create_synthetic_neighborhoods(self, grid_size: int = 4) -> gpd.GeoDataFrame:
        """
        Create synthetic neighborhood grid for analysis.
        
        Args:
            grid_size: Number of grid cells per dimension
        
        Returns:
            GeoDataFrame with synthetic neighborhoods
        """
        logger.info(f"Creating {grid_size}x{grid_size} synthetic neighborhood grid...")
        
        # Calculate grid cells
        lat_step = (self.bbox['north'] - self.bbox['south']) / grid_size
        lon_step = (self.bbox['east'] - self.bbox['west']) / grid_size
        
        neighborhoods = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                south = self.bbox['south'] + i * lat_step
                north = south + lat_step
                west = self.bbox['west'] + j * lon_step
                east = west + lon_step
                
                # Create polygon
                poly = Polygon([
                    (west, south),
                    (east, south),
                    (east, north),
                    (west, north)
                ])
                
                neighborhoods.append({
                    'name': f"Zone_{i}_{j}",
                    'grid_id': f"{i}_{j}",
                    'geometry': poly
                })
        
        gdf = gpd.GeoDataFrame(neighborhoods, crs='EPSG:4326')
        logger.info(f"Created {len(gdf)} synthetic neighborhoods")
        
        return gdf
    
    def fetch_gtfs_transit_stops(self, gtfs_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Parse GTFS data to extract transit stops.
        
        Args:
            gtfs_path: Path to GTFS zip file (if None, looks in GTFS_DATA_DIR)
        
        Returns:
            GeoDataFrame containing transit stop points
        """
        logger.info("Fetching transit stops from GTFS...")
        
        try:
            cache_path_gpkg = Path(CACHE_DIR) / 'wroclaw_transit_stops.gpkg'
            cache_path = Path(CACHE_DIR) / 'wroclaw_transit_stops.geojson'
            
            if cache_path_gpkg.exists():
                logger.info("Loading transit stops from GPKG cache...")
                self.transit_stops_gdf = gpd.read_file(cache_path_gpkg)
            elif cache_path.exists():
                logger.info("Loading transit stops from cache...")
                self.transit_stops_gdf = gpd.read_file(cache_path)
            else:
                # Look for GTFS file
                if gtfs_path is None:
                    gtfs_files = list(Path(GTFS_DATA_DIR).glob('*.zip'))
                    if gtfs_files:
                        gtfs_path = str(gtfs_files[0])
                    else:
                        logger.warning("No GTFS file found. Creating synthetic stops.")
                        self.transit_stops_gdf = self._create_synthetic_transit_stops()
                        return self.transit_stops_gdf
                
                # Parse GTFS using gtfs_kit
                try:
                    import gtfs_kit as gk
                    
                    feed = gk.read_feed(gtfs_path, dist_units='km')
                    stops_df = feed.stops
                    
                    # Create GeoDataFrame
                    geometry = [Point(xy) for xy in zip(stops_df['stop_lon'], stops_df['stop_lat'])]
                    self.transit_stops_gdf = gpd.GeoDataFrame(
                        stops_df,
                        geometry=geometry,
                        crs='EPSG:4326'
                    )
                    
                    # Filter to bbox
                    self.transit_stops_gdf = self.transit_stops_gdf.cx[
                        self.bbox['west']:self.bbox['east'],
                        self.bbox['south']:self.bbox['north']
                    ]
                    
                    # Sample if too many
                    if len(self.transit_stops_gdf) > 500:
                        self.transit_stops_gdf = self.transit_stops_gdf.sample(n=500, random_state=42)
                    
                except ImportError:
                    logger.warning("gtfs_kit not available. Creating synthetic stops.")
                    self.transit_stops_gdf = self._create_synthetic_transit_stops()
                except Exception as e:
                    logger.warning(f"Error parsing GTFS: {e}. Creating synthetic stops.")
                    self.transit_stops_gdf = self._create_synthetic_transit_stops()
                
                # Save to cache
                self.transit_stops_gdf.to_file(cache_path, driver='GeoJSON')
                logger.info(f"Saved transit stops to cache: {cache_path}")
            
            logger.info(f"Found {len(self.transit_stops_gdf)} transit stops")
            return self.transit_stops_gdf
            
        except Exception as e:
            logger.error(f"Error fetching transit stops: {e}")
            self.transit_stops_gdf = self._create_synthetic_transit_stops()
            return self.transit_stops_gdf
    
    def _create_synthetic_transit_stops(self, n_stops: int = 100) -> gpd.GeoDataFrame:
        """
        Create synthetic transit stops for testing when GTFS unavailable.
        
        Args:
            n_stops: Number of synthetic stops to create
        
        Returns:
            GeoDataFrame with synthetic transit stops
        """
        logger.info(f"Creating {n_stops} synthetic transit stops...")
        
        np.random.seed(42)
        
        # Random points within bbox
        lats = np.random.uniform(self.bbox['south'], self.bbox['north'], n_stops)
        lons = np.random.uniform(self.bbox['west'], self.bbox['east'], n_stops)
        
        stops = []
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            stops.append({
                'stop_id': f'synthetic_{i}',
                'stop_name': f'Synthetic Stop {i}',
                'geometry': Point(lon, lat)
            })
        
        gdf = gpd.GeoDataFrame(stops, crs='EPSG:4326')
        return gdf
    
    def calculate_neighborhood_features(self) -> pd.DataFrame:
        """
        Calculate walkability features for each neighborhood.
        
        Features include:
        - Sidewalk density (m/km²)
        - Crosswalk count
        - Average distance to nearest amenities
        - Average distance to nearest transit stops
        - Network connectivity metrics
        
        Returns:
            DataFrame with features per neighborhood
        """
        logger.info("Calculating neighborhood features...")
        
        features_list = []
        
        # Ensure data is loaded
        if self.neighborhoods_gdf is None or len(self.neighborhoods_gdf) == 0:
            logger.error("No neighborhoods loaded. Run fetch_neighborhoods() first.")
            return pd.DataFrame()
        
        # Project to metric CRS for distance calculations (EPSG:2180 for Poland)
        neighborhoods_proj = self.neighborhoods_gdf.to_crs(epsg=2180)
        
        # Process each neighborhood
        for idx, neighborhood in tqdm(self.neighborhoods_gdf.iterrows(), 
                                     total=len(self.neighborhoods_gdf),
                                     desc="Processing neighborhoods"):
            
            name = neighborhood.get('name', f'neighborhood_{idx}')
            geom = neighborhood.geometry
            geom_proj = neighborhoods_proj.loc[idx].geometry
            
            # Calculate area in km²
            area_km2 = geom_proj.area / 1_000_000
            
            # Get centroid for distance calculations
            centroid = geom.centroid
            
            # Initialize features
            features = {
                'neighborhood': name,
                'centroid_lat': centroid.y,
                'centroid_lon': centroid.x,
                'area_km2': area_km2
            }
            
            # 1. Sidewalk density
            if self.sidewalks_gdf is not None and len(self.sidewalks_gdf) > 0:
                sidewalks_in_area = self.sidewalks_gdf[self.sidewalks_gdf.intersects(geom)]
                sidewalks_proj = sidewalks_in_area.to_crs(epsg=2180)
                
                total_sidewalk_length = sidewalks_proj.length.sum()  # meters
                features['sidewalk_length_m'] = total_sidewalk_length
                features['sidewalk_density_m_per_km2'] = total_sidewalk_length / area_km2 if area_km2 > 0 else 0
            else:
                features['sidewalk_length_m'] = 0
                features['sidewalk_density_m_per_km2'] = 0
            
            # 2. Crosswalk count
            if self.crosswalks_gdf is not None and len(self.crosswalks_gdf) > 0:
                crosswalks_in_area = self.crosswalks_gdf[self.crosswalks_gdf.intersects(geom)]
                features['crosswalk_count'] = len(crosswalks_in_area)
                features['crosswalk_density_per_km2'] = len(crosswalks_in_area) / area_km2 if area_km2 > 0 else 0
            else:
                features['crosswalk_count'] = 0
                features['crosswalk_density_per_km2'] = 0
            
            # 3. Amenity distances
            if self.amenities_gdf is not None and len(self.amenities_gdf) > 0:
                amenity_distances = self._calculate_nearest_distances(
                    centroid, 
                    self.amenities_gdf, 
                    max_distance=MAX_AMENITY_DISTANCE
                )
                features['avg_amenity_distance_m'] = amenity_distances['mean']
                features['min_amenity_distance_m'] = amenity_distances['min']
                features['amenity_count_1km'] = amenity_distances['count']
            else:
                features['avg_amenity_distance_m'] = MAX_AMENITY_DISTANCE
                features['min_amenity_distance_m'] = MAX_AMENITY_DISTANCE
                features['amenity_count_1km'] = 0
            
            # 4. Transit stop distances
            if self.transit_stops_gdf is not None and len(self.transit_stops_gdf) > 0:
                transit_distances = self._calculate_nearest_distances(
                    centroid, 
                    self.transit_stops_gdf, 
                    max_distance=MAX_TRANSIT_DISTANCE
                )
                features['avg_transit_distance_m'] = transit_distances['mean']
                features['min_transit_distance_m'] = transit_distances['min']
                features['transit_count_500m'] = transit_distances['count']
            else:
                features['avg_transit_distance_m'] = MAX_TRANSIT_DISTANCE
                features['min_transit_distance_m'] = MAX_TRANSIT_DISTANCE
                features['transit_count_500m'] = 0
            
            # 5. Network metrics (simplified)
            features['network_connectivity'] = self._calculate_network_connectivity(centroid)
            
            features_list.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        logger.info(f"Calculated features for {len(features_df)} neighborhoods")
        logger.info(f"Features: {list(features_df.columns)}")
        
        return features_df
    
    def _calculate_nearest_distances(self, point: Point, gdf: gpd.GeoDataFrame, 
                                    max_distance: float, k: int = 5) -> Dict:
        """
        Calculate distances from a point to nearest features in GeoDataFrame.
        
        Args:
            point: Source point
            gdf: GeoDataFrame with target features
            max_distance: Maximum distance threshold (meters)
            k: Number of nearest neighbors to consider
        
        Returns:
            Dict with mean, min distances and count within threshold
        """
        # Calculate distances in meters (using geodesic approximation)
        distances = gdf.geometry.distance(point) * 111_000  # deg to meters (rough)
        
        # Get k nearest
        nearest_distances = distances.nsmallest(min(k, len(distances)))
        
        # Count within threshold
        count_within = (distances <= max_distance).sum()
        
        return {
            'mean': nearest_distances.mean() if len(nearest_distances) > 0 else max_distance,
            'min': nearest_distances.min() if len(nearest_distances) > 0 else max_distance,
            'count': count_within
        }
    
    def _calculate_network_connectivity(self, point: Point, radius: int = 500) -> float:
        """
        Calculate network connectivity around a point.
        
        Args:
            point: Center point
            radius: Radius in meters
        
        Returns:
            Connectivity score (0-1)
        """
        if self.pedestrian_graph is None:
            return 0.0
        
        try:
            # Get nearest node
            nearest_node = ox.distance.nearest_nodes(
                self.pedestrian_graph,
                point.x,
                point.y
            )
            
            # Count reachable nodes within radius (simplified metric)
            # For MVP, use degree of nearest node as proxy
            degree = self.pedestrian_graph.degree(nearest_node)
            
            # Normalize (typical range 1-6 for walking networks)
            connectivity = min(degree / 6.0, 1.0)
            
            return connectivity
            
        except Exception as e:
            logger.debug(f"Error calculating connectivity: {e}")
            return 0.5  # Default moderate connectivity
    
    def save_features(self, features_df: pd.DataFrame, filepath: str = None) -> None:
        """
        Save feature DataFrame to CSV.
        
        Args:
            features_df: DataFrame with neighborhood features
            filepath: Output file path (default: FEATURE_FILE from config)
        """
        if filepath is None:
            filepath = FEATURE_FILE
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(filepath, index=False)
        logger.info(f"Saved features to {filepath}")
    
    def run_full_pipeline(self, gtfs_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete data gathering and feature extraction pipeline.
        
        Args:
            gtfs_path: Optional path to GTFS file
        
        Returns:
            DataFrame with neighborhood features
        """
        logger.info("=" * 60)
        logger.info("Starting Wrocław Walkability Data Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Fetch all data
        self.fetch_pedestrian_network()
        self.fetch_sidewalks()
        self.fetch_crosswalks()
        self.fetch_amenities()
        self.fetch_neighborhoods()
        self.fetch_gtfs_transit_stops(gtfs_path)
        
        # Step 2: Calculate features
        features_df = self.calculate_neighborhood_features()
        
        # Step 3: Save features
        self.save_features(features_df)
        
        logger.info("=" * 60)
        logger.info("Data pipeline completed successfully!")
        logger.info(f"Features saved to: {FEATURE_FILE}")
        logger.info("=" * 60)
        
        return features_df


def main():
    """Main execution function."""
    print("Wrocław Walkability Analyzer - Data Gathering Module")
    print("=" * 60)
    
    # Initialize gatherer
    gatherer = WroclawDataGatherer()
    
    # Run full pipeline
    features_df = gatherer.run_full_pipeline()
    
    # Display summary
    print("\nFeature Summary:")
    print(features_df.describe())
    
    print("\nSample Data:")
    print(features_df.head())
    
    print("\n✓ Data gathering complete!")
    print(f"✓ {len(features_df)} neighborhoods processed")
    print(f"✓ {len(features_df.columns)} features calculated")


if __name__ == "__main__":
    main()
