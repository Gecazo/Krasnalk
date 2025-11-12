"""
Unified Wrocław Walkability Analyzer
====================================

All-in-one tool: Draw neighborhoods, analyze data, view results
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import geopandas as gpd
from shapely.geometry import shape
import json
import subprocess
import sys

# Import project modules
from config import WROCLAW_CENTER, FEATURE_FILE, OUTPUT_DIR, CACHE_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Wrocław Walkability Analyzer",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Persistence functions
TEMP_SAVE_PATH = Path('data/cache/neighborhoods_draft.json')

def load_draft():
    """Load draft neighborhoods from file."""
    if TEMP_SAVE_PATH.exists():
        try:
            with open(TEMP_SAVE_PATH, 'r') as f:
                data = json.load(f)
                return [{'name': item['name'], 'geometry': shape(item['geometry'])} 
                        for item in data]
        except:
            return []
    return []

def save_draft(neighborhoods):
    """Save draft neighborhoods to file."""
    TEMP_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [{'name': n['name'], 'geometry': n['geometry'].__geo_interface__} 
            for n in neighborhoods]
    with open(TEMP_SAVE_PATH, 'w') as f:
        json.dump(data, f)

@st.cache_data
def load_data():
    """Load features from file."""
    try:
        features_df = pd.read_csv(FEATURE_FILE)
        return features_df
    except FileNotFoundError:
        return None

# Initialize session state
if 'neighborhoods' not in st.session_state:
    st.session_state.neighborhoods = load_draft()
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'draw'  # 'draw' or 'analyze'

# Header
st.markdown('<div class="main-header">🚶 Wrocław Walkability Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Draw custom neighborhoods and analyze walkability metrics</div>', unsafe_allow_html=True)

# Mode selector
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🗺️ Draw Neighborhoods", use_container_width=True, 
                 type="primary" if st.session_state.view_mode == 'draw' else "secondary"):
        st.session_state.view_mode = 'draw'
        st.rerun()
with col2:
    if st.button("📊 View Results", use_container_width=True,
                 type="primary" if st.session_state.view_mode == 'analyze' else "secondary"):
        st.session_state.view_mode = 'analyze'
        st.rerun()
with col3:
    saved_count = len(st.session_state.neighborhoods)
    st.metric("Saved Areas", saved_count)

st.divider()

# ==================== DRAWING MODE ====================
if st.session_state.view_mode == 'draw':
    st.subheader("🗺️ Draw Custom Neighborhoods")
    
    # Create map with drawing tools
    m = folium.Map(
        location=WROCLAW_CENTER,
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add draw control
    draw = Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'polygon': True
        }
    )
    draw.add_to(m)
    
    # Add existing neighborhoods to map
    for idx, nbhd in enumerate(st.session_state.neighborhoods):
        folium.GeoJson(
            nbhd['geometry'],
            style_function=lambda x: {
                'fillColor': '#3186cc',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.3
            },
            tooltip=nbhd['name']
        ).add_to(m)
    
    # Display map
    col_map, col_list = st.columns([3, 1])
    
    with col_map:
        map_data = st_folium(m, width=None, height=600, key="unified_map")
    
    with col_list:
        st.markdown("### 📋 Neighborhoods")
        
        # Handle drawn shapes
        if map_data and map_data.get('last_active_drawing'):
            drawn = map_data['last_active_drawing']
            
            st.success("✅ Shape drawn!")
            name = st.text_input("Name this area:", key="nbhd_name_input")
            
            if st.button("💾 Save", key="save_btn"):
                if name:
                    try:
                        geom = shape(drawn['geometry'])
                        st.session_state.neighborhoods.append({
                            'name': name,
                            'geometry': geom
                        })
                        save_draft(st.session_state.neighborhoods)
                        st.success(f"✅ Saved: {name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("⚠️ Enter a name first!")
        
        # List existing neighborhoods
        if st.session_state.neighborhoods:
            for idx, nbhd in enumerate(st.session_state.neighborhoods):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"{idx+1}. {nbhd['name']}")
                with col_b:
                    if st.button("🗑️", key=f"del_{idx}"):
                        st.session_state.neighborhoods.pop(idx)
                        save_draft(st.session_state.neighborhoods)
                        st.rerun()
        else:
            st.info("No areas yet.\n\nDraw on the map!")
        
        st.divider()
        
        # Action buttons
        if st.button("✅ Analyze All", 
                     disabled=len(st.session_state.neighborhoods)==0, 
                     type="primary",
                     use_container_width=True):
            try:
                # Save neighborhoods
                gdf = gpd.GeoDataFrame(st.session_state.neighborhoods, crs='EPSG:4326')
                gdf['admin_level'] = 10
                output_path = Path('data/cache/wroclaw_neighborhoods.gpkg')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                gdf.to_file(output_path, driver='GPKG')
                
                st.success(f"✅ Saved {len(gdf)} areas!")
                
                # Run data gathering
                with st.spinner("Analyzing... This may take a few minutes."):
                    result = subprocess.run(
                        [sys.executable, 'data_gather.py'],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        cwd=Path.cwd()
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Analysis complete!")
                        st.balloons()
                        # Clear cache to load fresh data
                        load_data.clear()
                        # Switch to results view
                        st.session_state.view_mode = 'analyze'
                        st.rerun()
                    else:
                        st.error(f"Error:\n{result.stderr[:500]}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.neighborhoods = []
            save_draft(st.session_state.neighborhoods)
            st.rerun()

# ==================== RESULTS MODE ====================
else:
    st.subheader("📊 Walkability Analysis Results")
    
    features_df = load_data()
    
    if features_df is None or len(features_df) == 0:
        st.warning("⚠️ No data available. Please draw and analyze neighborhoods first.")
        if st.button("← Back to Drawing"):
            st.session_state.view_mode = 'draw'
            st.rerun()
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Neighborhoods", len(features_df))
        with col2:
            st.metric("Avg Area", f"{features_df['area_km2'].mean():.2f} km²")
        with col3:
            st.metric("Avg Sidewalk Density", f"{features_df['sidewalk_density_m_per_km2'].mean():.0f} m/km²")
        with col4:
            st.metric("Avg Amenities (1km)", f"{features_df['amenity_count_1km'].mean():.0f}")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📈 Feature Analysis", "📋 Data Table"])
        
        with tab1:
            # Create map with neighborhoods
            m = folium.Map(
                location=WROCLAW_CENTER,
                zoom_start=13,
                tiles='OpenStreetMap'
            )
            
            # Load neighborhood geometries
            neighborhoods_gdf = None
            cache_gpkg = Path(CACHE_DIR) / 'wroclaw_neighborhoods.gpkg'
            
            if cache_gpkg.exists():
                neighborhoods_gdf = gpd.read_file(cache_gpkg)
                merged = neighborhoods_gdf.merge(
                    features_df,
                    left_on='name',
                    right_on='neighborhood',
                    how='inner'
                )
                
                # Add polygons
                for idx, row in merged.iterrows():
                    popup_html = f"""
                    <div style="font-family: Arial; width: 250px;">
                        <h4 style="margin: 0;">{row['neighborhood']}</h4>
                        <hr style="margin: 5px 0;">
                        <p style="margin: 5px 0; font-size: 12px;">
                            <b>Area:</b> {row['area_km2']:.2f} km²<br>
                            <b>Sidewalk Density:</b> {row['sidewalk_density_m_per_km2']:.0f} m/km²<br>
                            <b>Crosswalks:</b> {row['crosswalk_count']:.0f}<br>
                            <b>Amenities (1km):</b> {row['amenity_count_1km']:.0f}<br>
                            <b>Transit Stops (500m):</b> {row['transit_count_500m']:.0f}
                        </p>
                    </div>
                    """
                    
                    # Color by sidewalk density
                    density = row['sidewalk_density_m_per_km2']
                    max_density = features_df['sidewalk_density_m_per_km2'].max()
                    
                    if max_density > 0:
                        normalized = density / max_density
                        if normalized >= 0.7:
                            color = '#28a745'
                        elif normalized >= 0.4:
                            color = '#5cb85c'
                        elif normalized >= 0.2:
                            color = '#ffc107'
                        else:
                            color = '#dc3545'
                    else:
                        color = '#6c757d'
                    
                    folium.GeoJson(
                        row['geometry'],
                        style_function=lambda x, color=color: {
                            'fillColor': color,
                            'color': 'black',
                            'weight': 2,
                            'fillOpacity': 0.5
                        },
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=row['neighborhood']
                    ).add_to(m)
            
            st_folium(m, width=None, height=600)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature distribution
                feature = st.selectbox(
                    "Select Feature",
                    ['sidewalk_density_m_per_km2', 'crosswalk_density_per_km2', 
                     'amenity_count_1km', 'transit_count_500m', 'network_connectivity']
                )
                
                fig = px.bar(
                    features_df,
                    x='neighborhood',
                    y=feature,
                    title=f'{feature.replace("_", " ").title()} by Neighborhood',
                    color=feature,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Correlation heatmap
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                numeric_cols = [c for c in numeric_cols if c not in ['centroid_lat', 'centroid_lon']]
                corr = features_df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr,
                    title='Feature Correlations',
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.dataframe(
                features_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = features_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "walkability_features.csv",
                "text/csv"
            )

# Instructions at bottom
st.divider()
with st.expander("💡 Instructions & Tips"):
    st.markdown("""
    ### How to Use This Tool
    
    **Step 1: Draw Neighborhoods**
    1. Switch to "🗺️ Draw Neighborhoods" mode
    2. Use the polygon tool (□) on the map to draw areas
    3. Click points to create corners, double-click to finish
    4. Name each area and click "💾 Save"
    5. Repeat for all areas you want to analyze
    
    **Step 2: Analyze**
    1. Click "✅ Analyze All" to process your neighborhoods
    2. Wait for the analysis to complete (may take a few minutes)
    3. Results will appear automatically
    
    **Step 3: View Results**
    1. Switch to "📊 View Results" mode
    2. Explore the interactive map with color-coded areas
    3. Check feature distributions and correlations
    4. View or download the raw data table
    
    ### Tips
    - **Zoom in** for more precise boundaries
    - Draw realistic neighborhood boundaries based on streets
    - Larger areas take longer to analyze
    - Popular Wrocław areas: Stare Miasto, Krzyki, Nadodrze, Śródmieście
    
    ### Walkability Metrics
    - **Sidewalk Density**: Length of sidewalks per km²
    - **Crosswalk Density**: Number of pedestrian crossings per km²
    - **Amenity Count**: Shops, services, parks within 1km
    - **Transit Accessibility**: Public transport stops within 500m
    - **Network Connectivity**: How well streets are connected
    """)
