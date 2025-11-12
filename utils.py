"""
Utility functions for Wrocław Walkability Analyzer.

Common helper functions used across the project.
"""

import logging
from typing import Tuple, List
import numpy as np
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Coordinates of first point (degrees)
        lat2, lon2: Coordinates of second point (degrees)
    
    Returns:
        Distance in meters
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    
    return distance


def calculate_walking_time(distance_m: float, speed_mps: float = 1.4) -> float:
    """
    Calculate walking time for a given distance.
    
    Args:
        distance_m: Distance in meters
        speed_mps: Walking speed in meters per second (default: 1.4 m/s)
    
    Returns:
        Time in minutes
    """
    if distance_m <= 0:
        return 0.0
    
    time_seconds = distance_m / speed_mps
    time_minutes = time_seconds / 60
    
    return time_minutes


def normalize_scores(scores: np.ndarray, min_score: float = 0, max_score: float = 100) -> np.ndarray:
    """
    Normalize scores to a specified range.
    
    Args:
        scores: Array of scores
        min_score: Minimum value for normalization
        max_score: Maximum value for normalization
    
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    # Min-max normalization
    score_min = scores.min()
    score_max = scores.max()
    
    if score_max == score_min:
        return np.full_like(scores, (min_score + max_score) / 2)
    
    normalized = (scores - score_min) / (score_max - score_min)
    normalized = normalized * (max_score - min_score) + min_score
    
    return normalized


def categorize_score(score: float) -> str:
    """
    Categorize a walkability score into a qualitative label.
    
    Args:
        score: Walkability score (0-100)
    
    Returns:
        Category label
    """
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Moderate"
    else:
        return "Low"


def calculate_area_km2(geometry, crs: str = 'EPSG:2180') -> float:
    """
    Calculate area of a geometry in square kilometers.
    
    Args:
        geometry: Shapely geometry object
        crs: Coordinate reference system for projection (default: EPSG:2180 for Poland)
    
    Returns:
        Area in km²
    """
    try:
        import geopandas as gpd
        
        # Create GeoSeries and project
        gs = gpd.GeoSeries([geometry], crs='EPSG:4326')
        gs_proj = gs.to_crs(crs)
        
        # Calculate area in m² and convert to km²
        area_m2 = gs_proj.area.iloc[0]
        area_km2 = area_m2 / 1_000_000
        
        return area_km2
        
    except Exception as e:
        logger.error(f"Error calculating area: {e}")
        return 0.0


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate if coordinates are within valid ranges.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        True if valid, False otherwise
    """
    if not (-90 <= lat <= 90):
        return False
    if not (-180 <= lon <= 180):
        return False
    return True


def format_distance(distance_m: float) -> str:
    """
    Format distance for display.
    
    Args:
        distance_m: Distance in meters
    
    Returns:
        Formatted string
    """
    if distance_m < 1000:
        return f"{distance_m:.0f} m"
    else:
        return f"{distance_m / 1000:.2f} km"


def calculate_density(length_or_count: float, area_km2: float) -> float:
    """
    Calculate density metric.
    
    Args:
        length_or_count: Total length (m) or count of features
        area_km2: Area in square kilometers
    
    Returns:
        Density value
    """
    if area_km2 <= 0:
        return 0.0
    
    return length_or_count / area_km2


def get_percentile_rank(value: float, all_values: List[float]) -> float:
    """
    Get percentile rank of a value in a distribution.
    
    Args:
        value: Target value
        all_values: List of all values
    
    Returns:
        Percentile (0-100)
    """
    if len(all_values) == 0:
        return 50.0
    
    percentile = (sum(v <= value for v in all_values) / len(all_values)) * 100
    
    return percentile


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
    
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    
    return numerator / denominator


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clip value to a specified range.
    
    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clipped value
    """
    return max(min_val, min(max_val, value))
