"""
Unit tests for utility functions.

Run with: pytest tests/test_utils.py
"""

import pytest
import numpy as np
from utils import (
    haversine_distance,
    calculate_walking_time,
    normalize_scores,
    categorize_score,
    validate_coordinates,
    format_distance,
    calculate_density,
    safe_divide,
    clip_value
)


class TestHaversineDistance:
    """Tests for haversine distance calculation."""
    
    def test_same_point(self):
        """Distance between same point should be 0."""
        distance = haversine_distance(51.1079, 17.0385, 51.1079, 17.0385)
        assert distance == pytest.approx(0, abs=1)
    
    def test_known_distance(self):
        """Test with known distance (Wrocław center to approx 1km north)."""
        # Roughly 1km north
        distance = haversine_distance(51.1079, 17.0385, 51.1169, 17.0385)
        assert 900 < distance < 1100  # Should be ~1km
    
    def test_symmetry(self):
        """Distance should be symmetric."""
        lat1, lon1 = 51.1079, 17.0385
        lat2, lon2 = 51.1169, 17.0485
        
        d1 = haversine_distance(lat1, lon1, lat2, lon2)
        d2 = haversine_distance(lat2, lon2, lat1, lon1)
        
        assert d1 == pytest.approx(d2, rel=1e-9)


class TestWalkingTime:
    """Tests for walking time calculation."""
    
    def test_zero_distance(self):
        """Zero distance should give zero time."""
        time = calculate_walking_time(0)
        assert time == 0.0
    
    def test_one_km(self):
        """1km at 1.4 m/s should be ~12 minutes."""
        time = calculate_walking_time(1000, speed_mps=1.4)
        expected = 1000 / 1.4 / 60  # ~11.9 minutes
        assert time == pytest.approx(expected, rel=0.01)
    
    def test_custom_speed(self):
        """Test with custom walking speed."""
        time = calculate_walking_time(500, speed_mps=2.0)
        expected = 500 / 2.0 / 60  # 4.17 minutes
        assert time == pytest.approx(expected, rel=0.01)


class TestNormalizeScores:
    """Tests for score normalization."""
    
    def test_basic_normalization(self):
        """Test basic 0-100 normalization."""
        scores = np.array([10, 20, 30, 40, 50])
        normalized = normalize_scores(scores, 0, 100)
        
        assert normalized.min() == 0
        assert normalized.max() == 100
        assert len(normalized) == len(scores)
    
    def test_empty_array(self):
        """Empty array should return empty."""
        scores = np.array([])
        normalized = normalize_scores(scores)
        assert len(normalized) == 0
    
    def test_uniform_scores(self):
        """Uniform scores should return middle value."""
        scores = np.array([50, 50, 50, 50])
        normalized = normalize_scores(scores, 0, 100)
        assert np.all(normalized == 50)


class TestCategorizeScore:
    """Tests for score categorization."""
    
    def test_excellent(self):
        """Score >= 80 should be Excellent."""
        assert categorize_score(100) == "Excellent"
        assert categorize_score(80) == "Excellent"
    
    def test_good(self):
        """Score 60-79 should be Good."""
        assert categorize_score(75) == "Good"
        assert categorize_score(60) == "Good"
    
    def test_moderate(self):
        """Score 40-59 should be Moderate."""
        assert categorize_score(50) == "Moderate"
        assert categorize_score(40) == "Moderate"
    
    def test_low(self):
        """Score < 40 should be Low."""
        assert categorize_score(30) == "Low"
        assert categorize_score(0) == "Low"


class TestValidateCoordinates:
    """Tests for coordinate validation."""
    
    def test_valid_wroclaw(self):
        """Wrocław coordinates should be valid."""
        assert validate_coordinates(51.1079, 17.0385) is True
    
    def test_invalid_latitude(self):
        """Invalid latitude should return False."""
        assert validate_coordinates(91, 17.0385) is False
        assert validate_coordinates(-91, 17.0385) is False
    
    def test_invalid_longitude(self):
        """Invalid longitude should return False."""
        assert validate_coordinates(51.1079, 181) is False
        assert validate_coordinates(51.1079, -181) is False
    
    def test_edge_cases(self):
        """Test edge case coordinates."""
        assert validate_coordinates(90, 180) is True
        assert validate_coordinates(-90, -180) is True


class TestFormatDistance:
    """Tests for distance formatting."""
    
    def test_meters(self):
        """Distances < 1km should be in meters."""
        assert format_distance(500) == "500 m"
        assert format_distance(999) == "999 m"
    
    def test_kilometers(self):
        """Distances >= 1km should be in km."""
        assert format_distance(1000) == "1.00 km"
        assert format_distance(1500) == "1.50 km"


class TestCalculateDensity:
    """Tests for density calculation."""
    
    def test_basic_density(self):
        """Test basic density calculation."""
        density = calculate_density(1000, 1.0)  # 1000m in 1 km²
        assert density == 1000.0
    
    def test_zero_area(self):
        """Zero area should return 0."""
        density = calculate_density(1000, 0)
        assert density == 0.0
    
    def test_negative_area(self):
        """Negative area should return 0."""
        density = calculate_density(1000, -1)
        assert density == 0.0


class TestSafeDivide:
    """Tests for safe division."""
    
    def test_normal_division(self):
        """Normal division should work."""
        result = safe_divide(10, 2)
        assert result == 5.0
    
    def test_zero_denominator(self):
        """Zero denominator should return default."""
        result = safe_divide(10, 0, default=999)
        assert result == 999
    
    def test_zero_numerator(self):
        """Zero numerator should return 0."""
        result = safe_divide(0, 5)
        assert result == 0.0


class TestClipValue:
    """Tests for value clipping."""
    
    def test_within_range(self):
        """Value within range should be unchanged."""
        assert clip_value(50, 0, 100) == 50
    
    def test_below_min(self):
        """Value below min should be clipped to min."""
        assert clip_value(-10, 0, 100) == 0
    
    def test_above_max(self):
        """Value above max should be clipped to max."""
        assert clip_value(150, 0, 100) == 100
    
    def test_edge_values(self):
        """Edge values should be valid."""
        assert clip_value(0, 0, 100) == 0
        assert clip_value(100, 0, 100) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
