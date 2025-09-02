import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Property Value Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    try:
        # Try different pickle loading methods
        scaler = None
        model = None
        
        # Method 1: Standard pickle loading
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('linear_regression_model.pkl', 'rb') as f:
                model = pickle.load(f)
            st.success("✅ Models loaded successfully!")
            return scaler, model
        except Exception as e1:
            print(f"Standard loading failed: {e1}")
                
            # Method 2: Try joblib loading (common for sklearn models)
            try:
                import joblib
                scaler = joblib.load('scaler.pkl')
                model = joblib.load('linear_regression_model.pkl')
                st.success("✅ Models loaded with joblib!")
                return scaler, model
            except Exception as e3:
                st.warning(f"Joblib loading failed: {e3}")
                
            raise Exception(f"All loading methods failed. Last error: {e1}")
            
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.markdown("""
        **Please ensure these files are in the same directory as your app:**
        - `scaler.pkl`
        - `linear_regression_model.pkl`
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        
        # Provide detailed troubleshooting
        st.markdown("""
        ## 🔧 Troubleshooting Steps:
        
        ### Common Solutions:
        1. **Python Version Mismatch**
           ```bash
           # Check your Python version
           python --version
           ```
           Make sure you're using the same Python version used to train the model.
        
        2. **Install Compatible Libraries**
           ```bash
           pip install pickle5
           pip install joblib
           pip install scikit-learn==1.3.0  # Match your training version
           ```
        
        3. **Re-save Your Models** (Recommended)
           ```python
           # In your training script, save with joblib instead:
           import joblib
           joblib.dump(scaler, 'scaler.pkl')
           joblib.dump(model, 'linear_regression_model.pkl')
           
           # Or use pickle with protocol 2 for better compatibility:
           import pickle
           with open('scaler.pkl', 'wb') as f:
               pickle.dump(scaler, f, protocol=2)
           with open('linear_regression_model.pkl', 'wb') as f:
               pickle.dump(model, f, protocol=2)
           ```
        """)
        
        if st.button("🎮 Create Demo Models"):
            create_demo_models()
            st.rerun()
        
        st.stop()

def create_demo_models():
    """Create simple demo models for testing"""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import joblib
        
        # Create dummy data
        np.random.seed(42)
        n_features = 16  # Number of features excluding target
        X_dummy = np.random.random((100, n_features))
        y_dummy = np.random.uniform(100000, 500000, 100)  # Property values
        
        # Create and train dummy models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)
        
        model = LinearRegression()
        model.fit(X_scaled, y_dummy)
        
        # Save with joblib for better compatibility
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(model, 'linear_regression_model.pkl')
        
        st.success("✅ Demo models created successfully!")
        st.info("ℹ️ These are dummy models for testing. Replace with your actual trained models.")
        
    except Exception as e:
        st.error(f"Failed to create demo models: {e}")

# Feature definitions based on your preprocessing
FEATURE_INFO = {
    'land_size_sqm': {
        'type': 'numeric', 
        'min': 100, 'max': 800, 'default': 200, 'step': 1, 
        'description': 'Land area in square meters'
    },
    'house_size_sqm': {
        'type': 'numeric', 
        'min': 50, 'max': 500, 'default': 180, 'step': 1, 
        'description': 'House area in square meters'
    },
    'no_of_rooms': {
        'type': 'numeric', 
        'min': 1, 'max': 8, 'default': 3, 'step': 1, 
        'description': 'Number of rooms'
    },
    'no_of_bathrooms': {
        'type': 'numeric', 
        'min': 1, 'max': 5, 'default': 2, 'step': 1, 
        'description': 'Number of bathrooms'
    },
    'large_living_room': {
        'type': 'binary', 
        'default': 0, 
        'description': 'Has large living room',
        'display_map': {0: 'No', 1: 'Yes'}
    },
    'parking_space': {
        'type': 'binary', 
        'default': 1, 
        'description': 'Has parking space',
        'display_map': {0: 'No', 1: 'Yes'}
    },
    'front_garden': {
        'type': 'binary', 
        'default': 1, 
        'description': 'Has front garden',
        'display_map': {0: 'No', 1: 'Yes'}
    },
    'swimming_pool': {
        'type': 'binary', 
        'default': 0, 
        'description': 'Has swimming pool',
        'display_map': {0: 'No', 1: 'Yes'}
    },
    'distance_to_school_km': {
        'type': 'numeric', 
        'min': 0.1, 'max': 20.0, 'default': 3.5, 'step': 0.1, 
        'description': 'Distance to nearest school (km)'
    },
    'wall_fence': {
        'type': 'binary', 
        'default': 1, 
        'description': 'Has wall/fence',
        'display_map': {0: 'No', 1: 'Yes'}
    },
    'house_age_in_years': {
        'type': 'numeric', 
        'min': 0, 'max': 50, 'default': 5, 'step': 1, 
        'description': 'Age of house in years'
    },
    'water_front': {
        'type': 'binary', 
        'default': 0, 
        'description': 'Waterfront property',
        'display_map': {0: 'No', 1: 'Yes'}
    },
    'distance_to_supermarket_km': {
        'type': 'numeric', 
        'min': 0.1, 'max': 15.0, 'default': 2.5, 'step': 0.1, 
        'description': 'Distance to supermarket (km)'
    },
    'crime_rate_index': {
        'type': 'numeric', 
        'min': 0.0, 'max': 50.0, 'default': 5.0, 'step': 0.1, 
        'description': 'Crime rate index (lower is better)'
    },
    'room_size': {
        'type': 'categorical', 
        'default': 1, 
        'description': 'Room size category',
        'options': [0, 1, 2, 3],
        'display_map': {0: 'Small', 1: 'Medium', 2: 'Large', 3: 'Extra Large'}
    },
    'house_to_land_ratio': {
        'type': 'calculated', 
        'description': 'House to land size ratio (calculated automatically)'
    }
}

def load_sample_properties():
    """Load sample properties based on your actual data"""
    return {
        'Sample Property 1': {
            'land_size_sqm': 201, 'house_size_sqm': 177, 'no_of_rooms': 3, 'no_of_bathrooms': 1,
            'large_living_room': 0, 'parking_space': 1, 'front_garden': 1, 'swimming_pool': 0,
            'distance_to_school_km': 3.31, 'wall_fence': 1, 'house_age_in_years': 10, 'water_front': 0,
            'distance_to_supermarket_km': 6.8, 'crime_rate_index': 0.9, 'room_size': 0
        },
        'Sample Property 2': {
            'land_size_sqm': 196, 'house_size_sqm': 182, 'no_of_rooms': 4, 'no_of_bathrooms': 3,
            'large_living_room': 1, 'parking_space': 1, 'front_garden': 0, 'swimming_pool': 1,
            'distance_to_school_km': 1.21, 'wall_fence': 1, 'house_age_in_years': 11, 'water_front': 0,
            'distance_to_supermarket_km': 4.1, 'crime_rate_index': 1.42, 'room_size': 1
        },
        'Sample Property 3': {
            'land_size_sqm': 198, 'house_size_sqm': 182, 'no_of_rooms': 4, 'no_of_bathrooms': 4,
            'large_living_room': 1, 'parking_space': 1, 'front_garden': 0, 'swimming_pool': 1,
            'distance_to_school_km': 15.9, 'wall_fence': 0, 'house_age_in_years': 20, 'water_front': 0,
            'distance_to_supermarket_km': 2.14, 'crime_rate_index': 4.12, 'room_size': 1
        }
    }

def generate_realistic_market_data(n_samples=200):
    """Generate realistic market data for comparison"""
    np.random.seed(42)
    market_data = []
    
    for i in range(n_samples):
        property_data = {
            'land_size_sqm': np.random.randint(150, 250),
            'house_size_sqm': np.random.randint(120, 220),
            'no_of_rooms': np.random.randint(2, 6),
            'no_of_bathrooms': np.random.randint(1, 5),
            'large_living_room': np.random.choice([0, 1], p=[0.6, 0.4]),
            'parking_space': np.random.choice([0, 1], p=[0.2, 0.8]),
            'front_garden': np.random.choice([0, 1], p=[0.4, 0.6]),
            'swimming_pool': np.random.choice([0, 1], p=[0.8, 0.2]),
            'distance_to_school_km': np.random.uniform(0.5, 15.0),
            'wall_fence': np.random.choice([0, 1], p=[0.3, 0.7]),
            'house_age_in_years': np.random.randint(0, 31),
            'water_front': np.random.choice([0, 1], p=[0.9, 0.1]),
            'distance_to_supermarket_km': np.random.uniform(0.5, 10.0),
            'crime_rate_index': np.random.uniform(0.5, 20.0),
            'room_size': np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])
        }
        
        # Calculate house_to_land_ratio
        property_data['house_to_land_ratio'] = property_data['house_size_sqm'] / property_data['land_size_sqm']
        
        market_data.append(property_data)
    
    return market_data

def main():
    st.title("🏠 Property Value Predictor")
    st.markdown("**Based on Trained Linear Regression Model**")
    
    # Quick info about data preprocessing
    with st.expander("ℹ️ Data Preprocessing Info"):
        st.markdown("""
        **Feature Mappings:**
        - **Binary Features**: Yes → 1, No → 0
        - **Room Size**: Small → 0, Medium → 1, Large → 2, Extra Large → 3
        - **Numeric Features**: Scaled using trained StandardScaler
        """)
    
    st.markdown("---")
    
    # Load models
    scaler, model = load_models()
    
    # Sidebar for inputs
    st.sidebar.header("Property Features")
    
    # Sample property loader
    st.sidebar.markdown("**Quick Start:**")
    sample_properties = load_sample_properties()
    selected_sample = st.sidebar.selectbox(
        "Load Sample Property:",
        ["Custom Input"] + list(sample_properties.keys()),
        help="Load pre-configured sample properties from your dataset"
    )
    
    st.sidebar.markdown("**Enter Property Details:**")
    
    # Collect user inputs
    user_inputs = {}
    
    # If a sample is selected, use those values as defaults
    if selected_sample != "Custom Input":
        sample_data = sample_properties[selected_sample]
    else:
        sample_data = {}
    
    # Create input fields for each feature
    for feature, info in FEATURE_INFO.items():
        if info['type'] == 'calculated':
            continue  # Skip calculated fields
            
        # Use sample data value if available, otherwise use default
        default_value = sample_data.get(feature, info['default'])
        
        if info['type'] == 'binary':
            # Create selectbox with user-friendly labels
            options = [0, 1]
            labels = [info['display_map'][0], info['display_map'][1]]
            
            user_inputs[feature] = st.sidebar.selectbox(
                f"{feature.replace('_', ' ').title()}",
                options=options,
                index=int(default_value),
                help=info['description'],
                key=f"{feature}_{selected_sample}",
                format_func=lambda x: info['display_map'][x]
            )
            
        elif info['type'] == 'categorical':
            # Create selectbox with user-friendly labels for room_size
            options = info['options']
            
            user_inputs[feature] = st.sidebar.selectbox(
                f"{feature.replace('_', ' ').title()}",
                options=options,
                index=options.index(int(default_value)) if int(default_value) in options else 0,
                help=info['description'],
                key=f"{feature}_{selected_sample}",
                format_func=lambda x: info['display_map'][x]
            )
            
        else:
            # Numeric inputs
            user_inputs[feature] = st.sidebar.number_input(
                f"{feature.replace('_', ' ').title()}",
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=float(default_value),
                step=float(info['step']),
                help=info['description'],
                key=f"{feature}_{selected_sample}"
            )
    
    # Calculate derived feature
    user_inputs['house_to_land_ratio'] = user_inputs['house_size_sqm'] / user_inputs['land_size_sqm']
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Property Summary")
        
        # Display input summary with user-friendly labels
        summary_data = []
        for feature, value in user_inputs.items():
            feature_info = FEATURE_INFO.get(feature, {})
            feature_name = feature.replace('_', ' ').title()
            
            if feature_info.get('type') == 'binary':
                display_value = feature_info['display_map'][value]
            elif feature_info.get('type') == 'categorical':
                display_value = feature_info['display_map'][value]
            else:
                if isinstance(value, float):
                    display_value = f"{value:,.2f}"
                else:
                    display_value = f"{value:,}"
            
            summary_data.append({"Feature": feature_name, "Value": display_value})
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🎯 Prediction")
        
        # Prepare data for prediction
        feature_columns = [col for col in FEATURE_INFO.keys() if FEATURE_INFO[col]['type'] != 'calculated']
        feature_columns.append('house_to_land_ratio')  # Add calculated feature
        
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[feature_columns]
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display prediction
        st.metric(
            label="Predicted Property Value",
            value=f"${prediction:,.2f}",
            delta=None
        )
        
        # Add confidence interval (approximate)
        confidence_margin = prediction * 0.15  # 15% margin
        st.write(f"**Estimated Range:** ${prediction-confidence_margin:,.2f} - ${prediction+confidence_margin:,.2f}")
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 Insights & Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Impact", "Property Comparison", "Market Position", "Feature Analysis"])
    
    with tab1:
        st.subheader("Feature Importance Analysis")
        
        if hasattr(model, 'coef_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Coefficient': model.coef_,
                'Abs_Coefficient': np.abs(model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10),
                x='Abs_Coefficient',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features",
                labels={'Abs_Coefficient': 'Absolute Coefficient Value'},
                color='Coefficient',
                color_continuous_scale='RdYlBu'
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab2:
        st.subheader("Property Comparison")
        
        typical_values = {feature: info['default'] for feature, info in FEATURE_INFO.items() if info['type'] != 'calculated'}
        typical_values['house_to_land_ratio'] = 0.88  # typical ratio
        
        comparison_data = []
        for feature in feature_columns:
            comparison_data.append({
                'Feature': feature.replace('_', ' ').title(),
                'Your Property': user_inputs[feature],
                'Market Average': typical_values.get(feature, user_inputs[feature])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Your Property',
            x=comparison_df['Feature'],
            y=comparison_df['Your Property'],
            marker_color='lightblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Market Average',
            x=comparison_df['Feature'],
            y=comparison_df['Market Average'],
            marker_color='orange'
        ))
        
        fig_comparison.update_layout(
            title='Your Property vs Market Average',
            xaxis_tickangle=-45,
            height=500,
            barmode='group'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab3:
        st.subheader("Market Position Analysis")
        
        # Generate market data and predictions
        market_data = generate_realistic_market_data(200)
        market_predictions = []
        
        for property_data in market_data:
            market_df = pd.DataFrame([property_data])
            market_df = market_df[feature_columns]
            market_scaled = scaler.transform(market_df)
            market_pred = model.predict(market_scaled)[0]
            market_predictions.append(market_pred)
        
        # Market position visualization
        fig_market = go.Figure()
        
        fig_market.add_trace(go.Histogram(
            x=market_predictions,
            nbinsx=25,
            name='Market Distribution',
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        fig_market.add_vline(
            x=prediction,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Your Property: ${prediction:,.0f}",
            annotation_position="top right"
        )
        
        fig_market.update_layout(
            title='Your Property Value vs Market Distribution',
            xaxis_title='Property Value ($)',
            yaxis_title='Number of Properties',
            height=500
        )
        
        st.plotly_chart(fig_market, use_container_width=True)
        
        # Market statistics
        percentile = (np.sum(np.array(market_predictions) < prediction) / len(market_predictions)) * 100
        market_median = np.median(market_predictions)
        market_mean = np.mean(market_predictions)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Your Percentile", f"{percentile:.1f}%")
        
        with col2:
            st.metric("Market Median", f"${market_median:,.0f}")
            
        with col3:
            st.metric("Market Average", f"${market_mean:,.0f}")
            
        with col4:
            difference = prediction - market_mean
            st.metric("vs Market Avg", f"${difference:,.0f}")
        
        # Market insights
        if percentile > 75:
            st.success("🏆 Your property is in the top 25% of the market!")
        elif percentile > 50:
            st.info("📈 Your property is above average in the market.")
        elif percentile > 25:
            st.warning("📊 Your property is below average but within normal range.")
        else:
            st.error("📉 Your property is in the bottom 25% of the market.")
    
    with tab4:
        st.subheader("Feature Analysis")
        
        st.write("**Sensitivity Analysis**: How property value changes with key features")
        
        # Analyze key numeric features
        sensitive_features = ['house_size_sqm', 'land_size_sqm', 'distance_to_school_km', 'house_age_in_years']
        
        for feature in sensitive_features[:2]:  # Show top 2
            feature_info = FEATURE_INFO[feature]
            min_val, max_val = feature_info['min'], feature_info['max']
            feature_range = np.linspace(min_val, max_val, 20)
            
            predictions_range = []
            for val in feature_range:
                temp_inputs = user_inputs.copy()
                temp_inputs[feature] = val
                # Recalculate ratio if land_size changes
                if feature == 'land_size_sqm':
                    temp_inputs['house_to_land_ratio'] = temp_inputs['house_size_sqm'] / val
                elif feature == 'house_size_sqm':
                    temp_inputs['house_to_land_ratio'] = val / temp_inputs['land_size_sqm']
                    
                temp_df = pd.DataFrame([temp_inputs])
                temp_df = temp_df[feature_columns]
                temp_scaled = scaler.transform(temp_df)
                pred = model.predict(temp_scaled)[0]
                predictions_range.append(pred)
            
            fig_sensitivity = go.Figure()
            fig_sensitivity.add_trace(go.Scatter(
                x=feature_range,
                y=predictions_range,
                mode='lines+markers',
                name=f'{feature.replace("_", " ").title()} Impact'
            ))
            
            fig_sensitivity.add_vline(
                x=user_inputs[feature],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Current: {user_inputs[feature]}"
            )
            
            fig_sensitivity.update_layout(
                title=f'Property Value Sensitivity to {feature.replace("_", " ").title()}',
                xaxis_title=feature.replace("_", " ").title(),
                yaxis_title='Predicted Property Value ($)',
                height=400
            )
            
            st.plotly_chart(fig_sensitivity, use_container_width=True)

if __name__ == "__main__":
    main()