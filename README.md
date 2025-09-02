# Property Value Predictor

A comprehensive machine learning application that predicts residential property values using various property characteristics and location factors. Built with Python, Streamlit, and scikit-learn.

## Project Overview

This project implements a Linear Regression model to predict property values based on 16 key features including property size, amenities, location, and neighborhood characteristics. The interactive web application provides real-time predictions with detailed analysis and market comparisons.

### Key Features

- **Interactive Web Interface**: Built with Streamlit for user-friendly property value predictions
- **Comprehensive Analysis**: Multiple visualization tabs including feature importance, market position, and sensitivity analysis
- **High Accuracy**: Achieves 98.8%+ R² score on both training and test datasets
- **Market Insights**: Compare properties against market averages and distributions
- **Sample Properties**: Pre-loaded sample properties for quick testing

## Technologies Used

- **Python 3.7+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **Web App**: Streamlit
- **Model Persistence**: joblib, pickle

## Project Structure

```
property-value-predictor/
│
├── app.py                              # Streamlit web application
├── data_analysis.ipynb                 # Data preprocessing and model training
├── linear_regression_model.pkl         # Trained model
├── scaler.pkl                          # Feature scaler
├── DataSet/
│   └── house_price_dataset_original_v2_with_categorical_features.csv
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn joblib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/property-value-predictor.git
cd property-value-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:PORT`

## Model Performance

- **Training R²**: 98.81%
- **Testing R²**: 98.85%
- **Training RMSE**: $9,852
- **Testing RMSE**: $9,604

The model demonstrates excellent performance with high accuracy and minimal overfitting.

## Features Used for Prediction

### Property Characteristics
- **Land Size** (sqm): Total land area
- **House Size** (sqm): Built-up area
- **Number of Rooms**: Total rooms in property
- **Number of Bathrooms**: Total bathrooms
- **House Age**: Age of property in years
- **House-to-Land Ratio**: Calculated feature

### Amenities (Binary Features)
- Large Living Room
- Parking Space
- Front Garden
- Swimming Pool
- Wall/Fence
- Waterfront Access

### Location Factors
- **Distance to School** (km): Proximity to nearest school
- **Distance to Supermarket** (km): Proximity to shopping
- **Crime Rate Index**: Neighborhood safety rating

### Property Classification
- **Room Size Category**: Small, Medium, Large, Extra Large

## Key Insights from Analysis

1. **Strong Correlations**: House size (r=0.96) and land size (r=0.97) show the strongest correlation with property values
2. **Location Matters**: Properties closer to schools and supermarkets tend to have higher values
3. **Amenities Impact**: Swimming pools and large living rooms significantly increase property values
4. **Size Categories**: Larger room categories correlate with higher property values
5. **Age Factor**: Newer properties generally command higher prices

## Using the Application

### Quick Start Options
1. **Load Sample Properties**: Choose from pre-configured sample properties
2. **Custom Input**: Enter your own property details
3. **Real-time Predictions**: Get instant value estimates with confidence intervals

### Analysis Features
- **Feature Impact**: View which features most influence property values
- **Property Comparison**: Compare your property against market averages
- **Market Position**: See how your property ranks in the market distribution
- **Sensitivity Analysis**: Understand how changes in key features affect value

## Data Preprocessing

The model includes comprehensive data preprocessing:

- **Text Cleaning**: Removal of units (sqm, km, Years) from numeric columns
- **Categorical Encoding**: Binary features (Yes/No → 1/0), Room sizes (Small/Medium/Large/Extra Large → 0/1/2/3)
- **Feature Scaling**: StandardScaler for optimal model performance
- **Derived Features**: House-to-land ratio calculation
- **Data Validation**: Duplicate removal and missing value handling

## Model Training Process

```python
# Key steps in model development
1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Train/test split (80/20)
5. StandardScaler fitting
6. Linear Regression training
7. Model evaluation and validation
8. Model persistence with joblib
```

## Requirements

Create a `requirements.txt` file:
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

## Future Enhancements

- [ ] Add more advanced ML models (Random Forest, XGBoost)
- [ ] Implement feature selection optimization
- [ ] Add geospatial analysis with mapping
- [ ] Include market trend predictions
- [ ] Add property image analysis capabilities
- [ ] Implement API endpoints for integration
- [ ] Add user authentication and saved predictions
- [ ] Include comparative market analysis (CMA)

