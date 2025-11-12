# Wrocław Walkability Analyzer - Setup Guide

## Quick Setup (5 minutes)

### 1. System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.10 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 500MB for dependencies + data

### 2. Installation Steps

#### Option A: Using pip (Recommended)

```bash
# Clone repository
git clone https://github.com/Gecazo/Krasnalk.git
cd Krasnalk

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
# Clone repository
git clone https://github.com/Gecazo/Krasnalk.git
cd Krasnalk

# Create conda environment
conda create -n walkability python=3.10
conda activate walkability

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test imports
python -c "import osmnx; import streamlit; print('✓ Installation successful!')"
```

### 4. Optional: Download GTFS Data

For real transit data (optional - synthetic data available):

1. Visit [Transit.land](https://www.transit.land/)
2. Search for "MPK Wrocław"
3. Download GTFS feed
4. Place `gtfs.zip` in `data/gtfs/` directory

### 5. Run the Pipeline

```bash
# Step 1: Gather data (3-5 min)
python data_gather.py

# Step 2: Train model (30-60 sec)
python ml_score.py

# Step 3: Launch app
streamlit run unified_app.py
```

## Troubleshooting

### Issue: Import errors for osmnx or geopandas

**Solution**: These packages require GDAL. Try:
```bash
# Windows
pip install pipwin
pipwin install gdal
pipwin install fiona

# Linux
sudo apt-get install gdal-bin libgdal-dev

# Mac
brew install gdal
```

### Issue: Slow data gathering

**Solution**: First run downloads ~100MB from OSM. Subsequent runs use cache (fast).

### Issue: Streamlit won't start

**Solution**: Ensure port 8501 is available:
```bash
streamlit run unified_app.py --server.port 8502
```

### Issue: Memory errors

**Solution**: Reduce data size in `config.py`:
```python
MAX_AMENITY_DISTANCE = 500  # Reduce from 1000
```

## Development Setup

For contributing or customization:

```bash
# Install dev dependencies
pip install -r requirements.txt pytest black flake8

# Run tests
pytest tests/ -v

# Format code
black *.py

# Check linting
flake8 *.py
```

## Docker Setup (Advanced)

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "unified_app.py"]
```

```bash
# Build and run
docker build -t walkability .
docker run -p 8501:8501 walkability
```

## Next Steps

1. Read the [README.md](README.md) for usage instructions
2. Check `config.py` to customize parameters
3. Explore the Streamlit dashboard at http://localhost:8501

## Support

- **Issues**: [GitHub Issues](https://github.com/Gecazo/Krasnalk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Gecazo/Krasnalk/discussions)
