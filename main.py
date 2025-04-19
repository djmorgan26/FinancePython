import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import streamlit as st
from io import StringIO, BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import shapiro, kstest, chi2_contingency, ttest_ind, f_oneway

# Set page configuration
st.set_page_config(
    page_title="Advanced Data Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Cache mechanism for optimizing data loading
@st.cache_data(ttl=3600)
def load_data(file_path, file_type="csv"):
    """
    Load data from a file and return it as a pandas DataFrame.
    
    Args:
        file_path: Path to the file or uploaded file object
        file_type: Type of file ('csv', 'excel', or 'json')
    
    Returns:
        pandas DataFrame or None if there was an error
    """
    try:
        if file_type == "csv":
            # Try to detect separator
            if isinstance(file_path, str):
                with open(file_path, 'r') as f:
                    sample = f.read(5000)  # Read first 5000 chars
            else:
                sample = file_path.getvalue()[:5000].decode('utf-8')
            
            if ',' in sample:
                sep = ','
            elif ';' in sample:
                sep = ';'
            elif '\t' in sample:
                sep = '\t'
            else:
                sep = None  # Let pandas guess
                
            df = pd.read_csv(file_path, sep=sep)
        elif file_type == "excel":
            df = pd.read_excel(file_path)
        elif file_type == "json":
            df = pd.read_json(file_path)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def clean_column_names(df):
    """Clean column names by replacing spaces with underscores and converting to lowercase"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df

def check_missing_values(df):
    """Check for missing values in the DataFrame"""
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })
    return missing_df[missing_df['Missing Values'] > 0]

def handle_missing_values(df, method="drop"):
    """Handle missing values in the DataFrame"""
    if method == "drop":
        return df.dropna()
    elif method == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df
    elif method == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df
    elif method == "zero":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(0)
        return df
    elif method == "mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
        return df
    else:
        return df

def detect_outliers(df, column, method="iqr"):
    """Detect outliers in a column using IQR or Z-score method"""
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    elif method == "zscore":
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > 3]
        return outliers, None, None

def get_feature_importance(df):
    """Calculate feature correlation with all other features"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
        
    correlations = numeric_df.corr()
    # Sum the absolute values of correlations for each feature
    importance = correlations.abs().sum().sort_values(ascending=False)
    return importance

def display_dashboard(df):
    """Display interactive dashboard for the dataset"""
    st.markdown('<div class="section-header">Dataset Dashboard</div>', unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    col4.metric("Duplicated Rows", f"{df.duplicated().sum():,}")
    
    # Data types
    dtypes = df.dtypes.value_counts().reset_index()
    dtypes.columns = ['Data Type', 'Count']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="section-header">Data Types</div>', unsafe_allow_html=True)
        st.dataframe(dtypes)
    
    with col2:
        st.markdown('<div class="section-header">Memory Usage</div>', unsafe_allow_html=True)
        memory_usage = df.memory_usage(deep=True)
        memory_usage = pd.DataFrame({
            'Column': ['Index'] + list(df.columns),
            'Memory (MB)': [memory_usage['Index'] / (1024 * 1024)] + [memory_usage[col] / (1024 * 1024) for col in df.columns]
        })
        memory_usage = memory_usage.sort_values('Memory (MB)', ascending=False)
        
        fig = px.bar(memory_usage, y='Column', x='Memory (MB)', orientation='h',
                  title="Memory Usage by Column (MB)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def perform_univariate_analysis(df, column):
    """Perform univariate analysis on a single column"""
    st.markdown(f'<div class="section-header">Analysis for: {column}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Statistics
        if pd.api.types.is_numeric_dtype(df[column]):
            stats = df[column].describe()
            stats_df = pd.DataFrame({
                'Statistic': stats.index,
                'Value': stats.values
            })
            st.dataframe(stats_df, use_container_width=True)
            
            # Additional statistics
            skewness = df[column].skew()
            kurtosis = df[column].kurtosis()
            st.write(f"Skewness: {skewness:.4f}")
            st.write(f"Kurtosis: {kurtosis:.4f}")
        else:
            # Categorical statistics
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = [column, 'Count']
            value_counts['Percentage'] = (value_counts['Count'] / len(df)) * 100
            st.dataframe(value_counts.head(10), use_container_width=True)
            
            if len(value_counts) > 10:
                st.info(f"Showing top 10 out of {len(value_counts)} unique values")
    
    with col2:
        # Visualization
        if pd.api.types.is_numeric_dtype(df[column]):
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=("Histogram", "Box Plot"),
                               vertical_spacing=0.15)
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=df[column], name='Histogram'),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(x=df[column], name='Box Plot'),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Bar chart for categorical
            value_counts = df[column].value_counts().reset_index().head(20)
            value_counts.columns = [column, 'Count']
            
            fig = px.bar(value_counts, x=column, y='Count',
                        title=f"Top 20 values for {column}")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def perform_bivariate_analysis(df, x_col, y_col):
    """Perform bivariate analysis on two columns"""
    st.markdown(f'<div class="section-header">Bivariate Analysis: {x_col} vs {y_col}</div>', unsafe_allow_html=True)
    
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
    
    if x_is_numeric and y_is_numeric:
        # Scatter plot for two numeric columns
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate correlation
            correlation = df[[x_col, y_col]].corr().iloc[0, 1]
            st.metric("Correlation coefficient", f"{correlation:.4f}")
            
            # Scatter plot with trend line
            fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                          title=f"{x_col} vs {y_col} with trend line")
            st.plotly_chart(fig, use_container_width=True)
            
    elif x_is_numeric and not y_is_numeric:
        # Box plot for numeric vs categorical
        fig = px.box(df, x=y_col, y=x_col, title=f"Distribution of {x_col} by {y_col}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    elif not x_is_numeric and y_is_numeric:
        # Box plot for categorical vs numeric
        fig = px.box(df, x=x_col, y=y_col, title=f"Distribution of {y_col} by {x_col}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Heatmap for two categorical columns
        contingency_table = pd.crosstab(df[x_col], df[y_col])
        fig = px.imshow(contingency_table, text_auto=True,
                       title=f"Contingency table: {x_col} vs {y_col}")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Chi-square test
        from scipy.stats import chi2_contingency
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        st.write(f"Chi-square statistic: {chi2:.4f}")
        st.write(f"p-value: {p:.4f}")
        if p < 0.05:
            st.write("There is a significant association between these variables (p < 0.05)")
        else:
            st.write("There is no significant association between these variables (p >= 0.05)")

def perform_time_series_analysis(df, date_col, value_col):
    """Perform time series analysis on date and value columns"""
    st.markdown(f'<div class="section-header">Time Series Analysis: {value_col} over {date_col}</div>', unsafe_allow_html=True)
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            st.error(f"Could not convert {date_col} to datetime: {str(e)}")
            return
    
    # Resample by different time periods
    time_periods = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }
    
    selected_period = st.selectbox("Select time period for aggregation", 
                                   list(time_periods.keys()))
    
    agg_method = st.selectbox("Select aggregation method", 
                              ["Mean", "Sum", "Min", "Max", "Count"])
    
    # Resample time series
    df_resampled = df.set_index(date_col)
    
    if agg_method == "Mean":
        df_resampled = df_resampled.resample(time_periods[selected_period])[value_col].mean().reset_index()
    elif agg_method == "Sum":
        df_resampled = df_resampled.resample(time_periods[selected_period])[value_col].sum().reset_index()
    elif agg_method == "Min":
        df_resampled = df_resampled.resample(time_periods[selected_period])[value_col].min().reset_index()
    elif agg_method == "Max":
        df_resampled = df_resampled.resample(time_periods[selected_period])[value_col].max().reset_index()
    elif agg_method == "Count":
        df_resampled = df_resampled.resample(time_periods[selected_period])[value_col].count().reset_index()
    
    # Line plot
    fig = px.line(df_resampled, x=date_col, y=value_col, 
                title=f"{agg_method} of {value_col} ({selected_period})")
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving averages
    st.markdown('<div class="section-header">Moving Averages</div>', unsafe_allow_html=True)
    
    ma_periods = st.select_slider("Select moving average periods", 
                                 options=[3, 7, 14, 30, 60, 90], 
                                 value=7)
    
    # Original series with moving average
    df_ma = df.sort_values(by=date_col).copy()
    df_ma[f"MA_{ma_periods}"] = df_ma[value_col].rolling(window=ma_periods).mean()
    
    fig = px.line(df_ma, x=date_col, y=[value_col, f"MA_{ma_periods}"],
                title=f"{value_col} with {ma_periods}-period Moving Average")
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonality and trend
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Ensure we have enough data points
        if len(df_resampled) > ma_periods * 2:
            decomposition_period = st.slider("Select period for seasonal decomposition", 
                                           min_value=2, 
                                           max_value=min(52, len(df_resampled)//2),
                                           value=min(12, len(df_resampled)//2))
            
            # Perform decomposition
            result = seasonal_decompose(df_resampled.set_index(date_col)[value_col], 
                                      model='additive', 
                                      period=decomposition_period)
            
            # Create plots
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            # Plot the components
            fig = make_subplots(rows=4, cols=1, 
                              subplot_titles=("Original", "Trend", "Seasonality", "Residuals"),
                              vertical_spacing=0.1,
                              shared_xaxes=True)
            
            # Original
            fig.add_trace(
                go.Scatter(x=df_resampled[date_col], y=df_resampled[value_col], name='Original'),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=df_resampled[date_col], y=trend, name='Trend'),
                row=2, col=1
            )
            
            # Seasonality
            fig.add_trace(
                go.Scatter(x=df_resampled[date_col], y=seasonal, name='Seasonality'),
                row=3, col=1
            )
            
            # Residuals
            fig.add_trace(
                go.Scatter(x=df_resampled[date_col], y=residual, name='Residuals'),
                row=4, col=1
            )
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not perform seasonal decomposition: {str(e)}")

def perform_clustering(df, selected_features, n_clusters=3):
    """Perform K-means clustering on selected features"""
    st.markdown('<div class="section-header">K-means Clustering</div>', unsafe_allow_html=True)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[selected_features])
    
    # Perform PCA for visualization if we have more than 2 features
    if len(selected_features) > 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Using PCA to visualize in 2D")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            
            st.write(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            
            # Show feature importance for PCA
            loadings = pd.DataFrame(
                pca.components_.T, 
                columns=['PC1', 'PC2'], 
                index=selected_features
            )
            
            # Plot PCA loadings
            fig = px.imshow(loadings, text_auto=True, aspect="auto",
                          title="PCA Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Perform clustering on the scaled data
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_data)
            pca_df['Cluster'] = df['Cluster']
            
            # Plot the clusters
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                            title="K-means Clustering Results (PCA projection)")
            
            # Add centroids
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            fig.add_trace(
                go.Scatter(
                    x=centroids_pca[:, 0],
                    y=centroids_pca[:, 1],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color='black',
                        line=dict(width=2)
                    ),
                    name='Centroids'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        # If we only have 1 or 2 features, we can visualize directly
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        
        if len(selected_features) == 2:
            # Direct visualization with 2 features
            fig = px.scatter(df, x=selected_features[0], y=selected_features[1], 
                           color='Cluster', title="K-means Clustering Results")
            
            # Add centroids
            fig.add_trace(
                go.Scatter(
                    x=kmeans.cluster_centers_[:, 0],
                    y=kmeans.cluster_centers_[:, 1],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color='black',
                        line=dict(width=2)
                    ),
                    name='Centroids'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 1D visualization
            fig = px.histogram(df, x=selected_features[0], color='Cluster',
                             title="Distribution by Cluster", barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
    
    # Cluster profiles
    st.markdown('<div class="section-header">Cluster Profiles</div>', unsafe_allow_html=True)
    cluster_profiles = df.groupby('Cluster')[selected_features].mean()
    
    # Radar chart for cluster profiles
    categories = selected_features
    fig = go.Figure()

    for cluster in cluster_profiles.index:
        values = cluster_profiles.loc[cluster].values.tolist()
        # Close the loop
        values.append(values[0])
        
        # Normalize for radar chart
        min_vals = df[selected_features].min()
        max_vals = df[selected_features].max()
        normalized_values = [(val - min_v) / (max_v - min_v) if max_v > min_v else 0.5 
                           for val, min_v, max_v in zip(values[:-1], min_vals, max_vals)]
        normalized_values.append(normalized_values[0])  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories + [categories[0]],  # Close the loop
            fill='toself',
            name=f'Cluster {cluster}'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Cluster Profiles (Normalized)"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Show cluster statistics
    st.dataframe(cluster_profiles, use_container_width=True)
    
    # Show cluster sizes
    cluster_sizes = df['Cluster'].value_counts().reset_index()
    cluster_sizes.columns = ['Cluster', 'Count']
    cluster_sizes['Percentage'] = cluster_sizes['Count'] / len(df) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(cluster_sizes, use_container_width=True)
    
    with col2:
        fig = px.pie(cluster_sizes, values='Count', names='Cluster',
                    title="Cluster Distribution")
        st.plotly_chart(fig, use_container_width=True)

def export_data(df, format="csv"):
    """Export the DataFrame to various formats"""
    st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
    
    if format == "csv":
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
    elif format == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        excel_data = output.getvalue()
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name="processed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif format == "json":
        json_str = df.to_json(orient="records")
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name="processed_data.json",
            mime="application/json"
        )

def main():
    st.markdown('<div class="main-header">📊 Advanced Data Analysis App</div>', unsafe_allow_html=True)
    
    # Create sidebar for navigation and options
    with st.sidebar:
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        app_mode = st.selectbox("Choose Mode", 
                              ["Data Upload and Preprocessing", 
                               "Exploratory Data Analysis",
                               "Statistical Analysis",
                               "Time Series Analysis",
                               "Clustering Analysis",
                               "Export Data"])
        
        st.markdown('<div class="section-header">Sample Datasets</div>', unsafe_allow_html=True)
        sample_data = st.selectbox(
            "Load sample dataset",
            ["None", "Iris", "Titanic", "Boston Housing", "Wine Quality"]
        )
    
    # Initialize session state for DataFrame if not already created
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.original_df = None
    
    # Load sample data if selected
    if sample_data != "None":
        if sample_data == "Iris":
            df = sns.load_dataset("iris")
        elif sample_data == "Titanic":
            df = sns.load_dataset("titanic")
        elif sample_data == "Boston Housing":
            from sklearn.datasets import load_boston
            try:
                boston = load_boston()
                df = pd.DataFrame(boston.data, columns=boston.feature_names)
                df['MEDV'] = boston.target
            except:
                # Fallback if Boston dataset is unavailable (removed from scikit-learn)
                from sklearn.datasets import fetch_california_housing
                housing = fetch_california_housing()
                df = pd.DataFrame(housing.data, columns=housing.feature_names)
                df['MEDV'] = housing.target
        elif sample_data == "Wine Quality":
            # URL to wine quality dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            try:
                df = pd.read_csv(url, sep=';')
            except:
                st.error("Could not load Wine Quality dataset from UCI repository")
                df = None
        
        if df is not None:
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.success(f"Loaded {sample_data} dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Data Upload and Preprocessing
    if app_mode == "Data Upload and Preprocessing":
        st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)
        
        file_option = st.radio("Select data source", ["Upload File", "Enter URL"])
        
        if file_option == "Upload File":
            file_type = st.radio("Select file type", ["CSV", "Excel", "JSON"])
            
            if url and url.strip():
                try:
                    if file_type == "CSV":
                        df = pd.read_csv(url)
                    elif file_type == "Excel":
                        df = pd.read_excel(url)
                    elif file_type == "JSON":
                        df = pd.read_json(url)
                    
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        st.success(f"Successfully loaded data from URL with {df.shape[0]} rows and {df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Error loading data from URL: {str(e)}")
        
        # Display data preview if available
        if st.session_state.df is not None:
            df = st.session_state.df
            
            st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.write("Memory Usage:", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                st.write("Data Types:")
                st.write(df.dtypes)
            
            # Data preprocessing options
            st.markdown('<div class="section-header">Data Preprocessing</div>', unsafe_allow_html=True)
            
            preprocessing_tabs = st.tabs(["Cleaning", "Missing Values", "Feature Engineering", "Filtering"])
            
            with preprocessing_tabs[0]:
                st.markdown("### Data Cleaning")
                
                if st.checkbox("Clean Column Names"):
                    df = clean_column_names(df)
                    st.success("Column names cleaned!")
                    st.write("New column names:", df.columns.tolist())
                
                if st.checkbox("Drop Duplicate Rows"):
                    original_rows = df.shape[0]
                    df = df.drop_duplicates()
                    new_rows = df.shape[0]
                    st.write(f"Removed {original_rows - new_rows} duplicate rows.")
                
                if st.checkbox("Convert Data Types"):
                    for col in df.columns:
                        col_type = st.selectbox(
                            f"Convert '{col}' from {df[col].dtype}",
                            ["Keep current", "string", "integer", "float", "categorical", "boolean", "datetime"],
                            key=f"convert_{col}"
                        )
                        
                        if col_type != "Keep current":
                            try:
                                if col_type == "string":
                                    df[col] = df[col].astype(str)
                                elif col_type == "integer":
                                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                                elif col_type == "float":
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                elif col_type == "categorical":
                                    df[col] = df[col].astype('category')
                                elif col_type == "boolean":
                                    df[col] = df[col].astype('boolean')
                                elif col_type == "datetime":
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                                
                                st.success(f"Converted '{col}' to {col_type}")
                            except Exception as e:
                                st.error(f"Error converting '{col}' to {col_type}: {str(e)}")
            
            with preprocessing_tabs[1]:
                st.markdown("### Missing Values")
                
                # Check for missing values
                missing_df = check_missing_values(df)
                if missing_df.empty:
                    st.success("No missing values found in the dataset!")
                else:
                    st.write("Missing values found:")
                    st.dataframe(missing_df)
                    
                    # Handle missing values
                    missing_method = st.selectbox(
                        "Select method to handle missing values",
                        ["Do nothing", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with zero"]
                    )
                    
                    if missing_method != "Do nothing":
                        if missing_method == "Drop rows":
                            df = handle_missing_values(df, "drop")
                        elif missing_method == "Fill with mean":
                            df = handle_missing_values(df, "mean")
                        elif missing_method == "Fill with median":
                            df = handle_missing_values(df, "median")
                        elif missing_method == "Fill with mode":
                            df = handle_missing_values(df, "mode")
                        elif missing_method == "Fill with zero":
                            df = handle_missing_values(df, "zero")
                        
                        st.success(f"Missing values handled using '{missing_method}'!")
                        
                        # Check for missing values again
                        missing_df = check_missing_values(df)
                        if missing_df.empty:
                            st.success("All missing values resolved!")
                        else:
                            st.warning("Some missing values remain:")
                            st.dataframe(missing_df)
            
            with preprocessing_tabs[2]:
                st.markdown("### Feature Engineering")
                
                if st.checkbox("Create new features"):
                    st.write("Select columns to create new features:")
                    
                    # Numeric columns for mathematical operations
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(num_cols) >= 2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            x_col = st.selectbox("Select first column", num_cols, key="feat_x_col")
                        
                        with col2:
                            y_col = st.selectbox("Select second column", num_cols, key="feat_y_col")
                        
                        operations = st.multiselect(
                            "Select operations to perform",
                            ["Sum", "Difference", "Product", "Ratio", "Mean", "Min", "Max"]
                        )
                        
                        if operations:
                            for op in operations:
                                if op == "Sum":
                                    new_col = f"{x_col}_plus_{y_col}"
                                    df[new_col] = df[x_col] + df[y_col]
                                elif op == "Difference":
                                    new_col = f"{x_col}_minus_{y_col}"
                                    df[new_col] = df[x_col] - df[y_col]
                                elif op == "Product":
                                    new_col = f"{x_col}_times_{y_col}"
                                    df[new_col] = df[x_col] * df[y_col]
                                elif op == "Ratio":
                                    new_col = f"{x_col}_div_{y_col}"
                                    df[new_col] = df[x_col] / df[y_col].replace(0, np.nan)
                                elif op == "Mean":
                                    new_col = f"mean_{x_col}_{y_col}"
                                    df[new_col] = (df[x_col] + df[y_col]) / 2
                                elif op == "Min":
                                    new_col = f"min_{x_col}_{y_col}"
                                    df[new_col] = df[[x_col, y_col]].min(axis=1)
                                elif op == "Max":
                                    new_col = f"max_{x_col}_{y_col}"
                                    df[new_col] = df[[x_col, y_col]].max(axis=1)
                                
                                st.success(f"Created new feature: '{new_col}'")
                    else:
                        st.warning("Need at least 2 numeric columns to create new features")
                
                # Binning numeric data
                if len(num_cols) > 0 and st.checkbox("Bin numeric data"):
                    col_to_bin = st.selectbox("Select column to bin", num_cols)
                    num_bins = st.slider("Number of bins", min_value=2, max_value=20, value=5)
                    
                    bin_method = st.radio("Binning method", ["Equal width", "Equal frequency", "Custom"])
                    
                    if bin_method == "Equal width":
                        df[f"{col_to_bin}_binned"] = pd.cut(df[col_to_bin], bins=num_bins)
                        st.success(f"Created binned feature: '{col_to_bin}_binned' using equal width bins")
                    
                    elif bin_method == "Equal frequency":
                        df[f"{col_to_bin}_binned"] = pd.qcut(df[col_to_bin], q=num_bins, duplicates='drop')
                        st.success(f"Created binned feature: '{col_to_bin}_binned' using equal frequency bins")
                    
                    elif bin_method == "Custom":
                        min_val = float(df[col_to_bin].min())
                        max_val = float(df[col_to_bin].max())
                        
                        custom_bins = []
                        for i in range(num_bins + 1):
                            bin_val = st.number_input(
                                f"Bin edge {i}", 
                                min_value=min_val,
                                max_value=max_val,
                                value=min_val + (max_val - min_val) * i / num_bins
                            )
                            custom_bins.append(bin_val)
                        
                        if st.button("Create custom bins"):
                            df[f"{col_to_bin}_binned"] = pd.cut(df[col_to_bin], bins=custom_bins)
                            st.success(f"Created binned feature: '{col_to_bin}_binned' using custom bins")
                
                # One-hot encoding
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if cat_cols and st.checkbox("Apply one-hot encoding"):
                    cols_to_encode = st.multiselect("Select columns to encode", cat_cols)
                    
                    if cols_to_encode:
                        encoded_df = pd.get_dummies(df[cols_to_encode], prefix=cols_to_encode)
                        df = pd.concat([df, encoded_df], axis=1)
                        st.success(f"One-hot encoded {len(cols_to_encode)} column(s), created {encoded_df.shape[1]} new features")
            
            with preprocessing_tabs[3]:
                st.markdown("### Filtering Data")
                
                if st.checkbox("Filter by conditions"):
                    filter_col = st.selectbox("Select column to filter", df.columns)
                    
                    if pd.api.types.is_numeric_dtype(df[filter_col]):
                        filter_type = st.selectbox(
                            "Filter type", 
                            ["Greater than", "Less than", "Equal to", "Between", "Not equal to"]
                        )
                        
                        if filter_type in ["Greater than", "Less than", "Equal to", "Not equal to"]:
                            filter_value = st.number_input("Filter value", value=float(df[filter_col].mean()))
                            
                            if st.button("Apply numeric filter"):
                                original_rows = df.shape[0]
                                if filter_type == "Greater than":
                                    df = df[df[filter_col] > filter_value]
                                elif filter_type == "Less than":
                                    df = df[df[filter_col] < filter_value]
                                elif filter_type == "Equal to":
                                    df = df[df[filter_col] == filter_value]
                                elif filter_type == "Not equal to":
                                    df = df[df[filter_col] != filter_value]
                                
                                new_rows = df.shape[0]
                                st.write(f"Filtered from {original_rows} to {new_rows} rows.")
                        
                        elif filter_type == "Between":
                            min_val, max_val = st.slider(
                                "Range",
                                float(df[filter_col].min()),
                                float(df[filter_col].max()),
                                (float(df[filter_col].quantile(0.25)), float(df[filter_col].quantile(0.75)))
                            )
                            
                            if st.button("Apply range filter"):
                                original_rows = df.shape[0]
                                df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
                                new_rows = df.shape[0]
                                st.write(f"Filtered from {original_rows} to {new_rows} rows.")
                    
                    else:  # Categorical column
                        unique_values = df[filter_col].unique().tolist()
                        selected_values = st.multiselect("Select values to keep", unique_values)
                        
                        if selected_values and st.button("Apply categorical filter"):
                            original_rows = df.shape[0]
                            df = df[df[filter_col].isin(selected_values)]
                            new_rows = df.shape[0]
                            st.write(f"Filtered from {original_rows} to {new_rows} rows.")
                
                # Detect and filter outliers
                if st.checkbox("Detect and remove outliers"):
                    outlier_col = st.selectbox("Select column to check for outliers", 
                                             df.select_dtypes(include=[np.number]).columns)
                    
                    outlier_method = st.radio("Outlier detection method", ["IQR", "Z-Score"])
                    
                    outliers, lower_bound, upper_bound = detect_outliers(df, outlier_col, outlier_method.lower())
                    
                    if not outliers.empty:
                        st.write(f"Found {len(outliers)} outliers in column '{outlier_col}'")
                        
                        if outlier_method == "IQR":
                            st.write(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
                        
                        st.dataframe(outliers.head(10) if len(outliers) > 10 else outliers)
                        
                        if st.button("Remove outliers"):
                            original_rows = df.shape[0]
                            if outlier_method == "IQR":
                                df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                            else:  # Z-Score
                                z_scores = np.abs((df[outlier_col] - df[outlier_col].mean()) / df[outlier_col].std())
                                df = df[z_scores <= 3]
                            
                            new_rows = df.shape[0]
                            st.write(f"Removed {original_rows - new_rows} outliers.")
                
                # Random sampling
                if st.checkbox("Take a random sample"):
                    sample_size = st.slider("Sample size", 
                                          min_value=1, 
                                          max_value=df.shape[0],
                                          value=min(1000, df.shape[0]))
                    
                    sample_method = st.radio("Sampling method", ["Random", "Stratified"])
                    
                    if sample_method == "Random":
                        if st.button("Take random sample"):
                            df = df.sample(n=sample_size, random_state=42)
                            st.success(f"Sample of {sample_size} rows created")
                    
                    else:  # Stratified
                        strat_col = st.selectbox("Select column to stratify by", df.columns)
                        if st.button("Take stratified sample"):
                            try:
                                # Create sample
                                sample_pct = sample_size / len(df)
                                df = df.groupby(strat_col, group_keys=False).apply(
                                    lambda x: x.sample(max(int(len(x) * sample_pct), 1))
                                )
                                st.success(f"Stratified sample of approximately {len(df)} rows created")
                            except Exception as e:
                                st.error(f"Error creating stratified sample: {str(e)}")
            
            # Save preprocessed data
            st.session_state.df = df
            
            st.markdown('<div class="section-header">Save Preprocessed Data</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Changes", key="save_preprocessing"):
                    st.session_state.df = df
                    st.success("Changes saved! You can now proceed to analysis.")
            
            with col2:
                if st.button("Reset to Original Data", key="reset_preprocessing"):
                    df = st.session_state.original_df.copy()
                    st.session_state.df = df
                    st.success("Data reset to original state.")
    
    # Exploratory Data Analysis (EDA)
    elif app_mode == "Exploratory Data Analysis":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload and Preprocessing' tab first.")
        else:
            df = st.session_state.df
            
            # Display the dashboard overview
            display_dashboard(df)
            
            st.markdown('<div class="section-header">Exploratory Analysis</div>', unsafe_allow_html=True)
            
            eda_tabs = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
            
            with eda_tabs[0]:
                st.markdown("### Univariate Analysis")
                
                # Feature selection for univariate analysis
                selected_column = st.selectbox("Select column to analyze", df.columns)
                
                # Perform univariate analysis
                perform_univariate_analysis(df, selected_column)
                
                # Detect outliers for numeric columns
                if pd.api.types.is_numeric_dtype(df[selected_column]):
                    if st.checkbox("Detect outliers"):
                        outlier_method = st.radio("Detection method", ["IQR", "Z-Score"])
                        outliers, lower_bound, upper_bound = detect_outliers(
                            df, selected_column, method=outlier_method.lower())
                        
                        if not outliers.empty:
                            st.write(f"Found {len(outliers)} outliers out of {len(df)} values")
                            
                            if outlier_method == "IQR":
                                st.write(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
                            
                            # Show outliers
                            st.dataframe(outliers.head(10) if len(outliers) > 10 else outliers)
                        else:
                            st.write("No outliers detected!")
            
            with eda_tabs[1]:
                st.markdown("### Bivariate Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    x_column = st.selectbox("Select first column", df.columns, key="bi_x_col")
                
                with col2:
                    # Filter out the already selected column
                    available_cols = [col for col in df.columns if col != x_column]
                    y_column = st.selectbox("Select second column", available_cols, key="bi_y_col")
                
                # Perform bivariate analysis
                perform_bivariate_analysis(df, x_column, y_column)
            
            with eda_tabs[2]:
                st.markdown("### Multivariate Analysis")
                
                # Feature selection for multivariate analysis
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 3:
                    selected_features = st.multiselect(
                        "Select features for multivariate analysis",
                        numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))]
                    )
                    
                    if selected_features and len(selected_features) >= 2:
                        # Correlation heatmap
                        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
                        
                        corr = df[selected_features].corr()
                        
                        # Create a mask for the upper triangle
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        
                        # Generate heatmap with Plotly
                        fig = px.imshow(corr,
                                       x=corr.columns,
                                       y=corr.columns,
                                       color_continuous_scale='RdBu_r',
                                       zmin=-1, zmax=1,
                                       text_auto=True)
                        fig.update_layout(title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Scatter matrix
                        st.markdown('<div class="section-header">Scatter Matrix</div>', unsafe_allow_html=True)
                        
                        if len(selected_features) <= 6:  # Limit to 6 features for better visualization
                            fig = px.scatter_matrix(
                                df, 
                                dimensions=selected_features,
                                title="Scatter Matrix"
                            )
                            fig.update_layout(height=800)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Too many features selected for scatter matrix. Please select 6 or fewer features.")
                        
                        # Principal Component Analysis
                        if len(selected_features) >= 3 and st.checkbox("Perform PCA"):
                            st.markdown('<div class="section-header">Principal Component Analysis</div>', unsafe_allow_html=True)
                            
                            n_components = st.slider("Number of components", 
                                                   min_value=2, 
                                                   max_value=min(len(selected_features), 10),
                                                   value=min(3, len(selected_features)))
                            
                            # Standardize the data
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(df[selected_features])
                            
                            # Perform PCA
                            pca = PCA(n_components=n_components)
                            pca_result = pca.fit_transform(scaled_data)
                            
                            # Create DataFrame with PCA results
                            pca_df = pd.DataFrame(
                                data=pca_result, 
                                columns=[f'PC{i+1}' for i in range(n_components)]
                            )
                            
                            # Explained variance
                            explained_var = pca.explained_variance_ratio_
                            
                            # Plot explained variance
                            fig = px.bar(
                                x=[f'PC{i+1}' for i in range(n_components)],
                                y=explained_var,
                                labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
                                title="Explained Variance by Principal Component"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Plot PCA results
                            if n_components >= 2:
                                fig = px.scatter(
                                    pca_df, x='PC1', y='PC2',
                                    title="PCA: First Two Principal Components"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show loadings
                            loadings = pd.DataFrame(
                                pca.components_.T, 
                                columns=[f'PC{i+1}' for i in range(n_components)], 
                                index=selected_features
                            )
                            
                            st.markdown('<div class="section-header">PCA Feature Importance</div>', unsafe_allow_html=True)
                            st.dataframe(loadings, use_container_width=True)
                            
                            # Plot loadings
                            fig = px.imshow(loadings, text_auto=True, aspect="auto",
                                          title="PCA Loadings")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 features for multivariate analysis")
                else:
                    st.warning("Multivariate analysis requires at least 3 numeric columns")
    
    # Statistical Analysis
    elif app_mode == "Statistical Analysis":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload and Preprocessing' tab first.")
        else:
            df = st.session_state.df
            
            st.markdown('<div class="section-header">Statistical Analysis</div>', unsafe_allow_html=True)
            
            stat_tabs = st.tabs(["Descriptive Stats", "Group Comparison", "Correlation Analysis", "Distribution Tests"])
            
            with stat_tabs[0]:
                st.markdown("### Descriptive Statistics")
                
                # Select columns for analysis
                selected_cols = st.multiselect(
                    "Select columns for descriptive statistics",
                    df.columns,
                    default=df.select_dtypes(include=[np.number]).columns.tolist()[:5]
                )
                
                if selected_cols:
                    # Calculate descriptive stats
                    stats_df = df[selected_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
                    
                    # Additional statistics for numeric columns
                    for col in selected_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            stats_df.loc['skew', col] = df[col].skew()
                            stats_df.loc['kurtosis', col] = df[col].kurtosis()
                            stats_df.loc['iqr', col] = stats_df.loc['75%', col] - stats_df.loc['25%', col]
                            stats_df.loc['cv', col] = stats_df.loc['std', col] / stats_df.loc['mean', col] * 100 if stats_df.loc['mean', col] != 0 else np.nan
                    
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Display statistics visualizations
                    st.markdown('<div class="section-header">Statistics Visualization</div>', unsafe_allow_html=True)
                    
                    # Create box plots for numeric columns
                    numeric_cols = [col for col in selected_cols if pd.api.types.is_numeric_dtype(df[col])]
                    
                    if numeric_cols:
                        fig = go.Figure()
                        
                        for col in numeric_cols:
                            # Normalize data for better comparison
                            normalized_data = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                            fig.add_trace(go.Box(y=normalized_data, name=col))
                        
                        fig.update_layout(
                            title="Normalized Box Plots (Min-Max Scaling)",
                            yaxis_title="Normalized Value",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create violin plots
                        fig = go.Figure()
                        
                        for col in numeric_cols:
                            # Normalize data for better comparison
                            normalized_data = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                            fig.add_trace(go.Violin(y=normalized_data, name=col, box_visible=True, meanline_visible=True))
                        
                        fig.update_layout(
                            title="Normalized Violin Plots (Min-Max Scaling)",
                            yaxis_title="Normalized Value",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with stat_tabs[1]:
                st.markdown("### Group Comparison")
                
                # Select grouping and target variables
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if categorical_cols and numeric_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        group_col = st.selectbox("Select grouping variable", categorical_cols)
                    
                    with col2:
                        target_col = st.selectbox("Select target variable", numeric_cols)
                    
                    # Get group statistics
                    group_stats = df.groupby(group_col)[target_col].agg(['count', 'mean', 'std', 'min', 'median', 'max']).reset_index()
                    
                    # Add 95% confidence interval
                    def ci_95(x):
                        return 1.96 * x.std() / np.sqrt(x.count())
                    
                    ci = df.groupby(group_col)[target_col].agg(ci_95).reset_index()
                    ci.columns = [group_col, 'ci_95']
                    
                    group_stats = pd.merge(group_stats, ci, on=group_col)
                    group_stats['lower_ci'] = group_stats['mean'] - group_stats['ci_95']
                    group_stats['upper_ci'] = group_stats['mean'] + group_stats['ci_95']
                    
                    st.dataframe(group_stats, use_container_width=True)
                    
                    # Create visualizations
                    fig = make_subplots(rows=1, cols=2, 
                                       subplot_titles=("Group Means with 95% CI", "Group Box Plots"),
                                       specs=[[{"type": "bar"}, {"type": "box"}]])
                    
                    # Bar chart with error bars
                    fig.add_trace(
                        go.Bar(
                            x=group_stats[group_col],
                            y=group_stats['mean'],
                            error_y=dict(
                                type='data',
                                array=group_stats['ci_95'],
                                visible=True
                            ),
                            name="Mean with 95% CI"
                        ),
                        row=1, col=1
                    )
                    
                    # Box plots by group
                    for group in df[group_col].unique():
                        fig.add_trace(
                            go.Box(
                                y=df[df[group_col] == group][target_col],
                                name=str(group)
                            ),
                            row=1, col=2
                        )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical tests for group differences
                    st.markdown('<div class="section-header">Statistical Tests</div>', unsafe_allow_html=True)
                    
                    n_groups = df[group_col].nunique()
                    
                    if n_groups == 2:
                        st.write("#### Two-Sample t-test")
                        
                        try:
                            from scipy.stats import ttest_ind
                            
                            groups = df[group_col].unique()
                            group1 = df[df[group_col] == groups[0]][target_col].dropna()
                            group2 = df[df[group_col] == groups[1]][target_col].dropna()
                            
                            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
                            
                            st.write(f"T-statistic: {t_stat:.4f}")
                            st.write(f"P-value: {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.write("Result: There is a significant difference between the two groups (p < 0.05)")
                            else:
                                st.write("Result: There is no significant difference between the two groups (p >= 0.05)")
                        
                        except Exception as e:
                            st.error(f"Error performing t-test: {str(e)}")
                    
                    elif n_groups > 2:
                        st.write("#### One-way ANOVA")
                        
                        try:
                            from scipy.stats import f_oneway
                            
                            groups = [df[df[group_col] == group][target_col].dropna() for group in df[group_col].unique()]
                            f_stat, p_val = f_oneway(*groups)
                            
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"P-value: {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.write("Result: There is a significant difference between at least two groups (p < 0.05)")
                            else:
                                st.write("Result: There is no significant difference between the groups (p >= 0.05)")
                        
                        except Exception as e:
                            st.error(f"Error performing ANOVA: {str(e)}")
                else:
                    st.warning("Group comparison requires at least one categorical and one numeric column")
            
            with stat_tabs[2]:
                st.markdown("### Correlation Analysis")
                
                # Select numeric columns for correlation analysis
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    selected_cols = st.multiselect(
                        "Select columns for correlation analysis",
                        numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))]
                    )
                    
                    if len(selected_cols) >= 2:
                        corr_method = st.radio(
                            "Correlation method",
                            ["Pearson", "Spearman", "Kendall"]
                        )
                        
                        # Calculate correlation matrix
                        if corr_method == "Pearson":
                            corr = df[selected_cols].corr(method='pearson')
                        elif corr_method == "Spearman":
                            corr = df[selected_cols].corr(method='spearman')
                        else:  # Kendall
                            corr = df[selected_cols].corr(method='kendall')
                        
                        # Display correlation matrix as heatmap
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title=f"{corr_method} Correlation Matrix"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Strongest correlations
                        st.markdown('<div class="section-header">Strongest Correlations</div>', unsafe_allow_html=True)
                        
                        # Get correlations in pair format
                        corr_pairs = []
                        for i in range(len(selected_cols)):
                            for j in range(i+1, len(selected_cols)):
                                corr_pairs.append({
                                    'Variable 1': selected_cols[i],
                                    'Variable 2': selected_cols[j],
                                    'Correlation': corr.iloc[i, j]
                                })
                        
                        # Convert to DataFrame and sort
                        corr_df = pd.DataFrame(corr_pairs)
                        corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
                        corr_df = corr_df.sort_values('Abs Correlation', ascending=False).drop('Abs Correlation', axis=1)
                        
                        st.dataframe(corr_df, use_container_width=True)
                        
                        # Select pair for detailed analysis
                        if st.checkbox("Detailed pair analysis"):
                            # Select pair to analyze
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                x_col = st.selectbox("Select first variable", selected_cols, key="corr_x_var")
                            
                            with col2:
                                y_col = st.selectbox("Select second variable", 
                                                   [col for col in selected_cols if col != x_col], 
                                                   key="corr_y_var")
                            
                            # Calculate correlation for the pair
                            if corr_method == "Pearson":
                                corr_val, p_val = pd.Series(df[x_col]).corr(pd.Series(df[y_col]), method='pearson'), 0  # Placeholder for p-value
                            elif corr_method == "Spearman":
                                from scipy.stats import spearmanr
                                corr_val, p_val = spearmanr(df[x_col].dropna(), df[y_col].dropna())
                            else:  # Kendall
                                from scipy.stats import kendalltau
                                corr_val, p_val = kendalltau(df[x_col].dropna(), df[y_col].dropna())
                            
                            # Display correlation value
                            st.metric(f"{corr_method} Correlation", f"{corr_val:.4f}")
                            st.write(f"p-value: {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.write("This correlation is statistically significant (p < 0.05)")
                            else:
                                st.write("This correlation is not statistically significant (p >= 0.05)")
                            
                            # Scatter plot with trend line
                            fig = px.scatter(
                                df, x=x_col, y=y_col, 
                                trendline="ols", 
                                trendline_color_override="red",
                                title=f"Scatter Plot with Trend Line: {x_col} vs {y_col}"
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 columns for correlation analysis")
                else:
                    st.warning("Correlation analysis requires at least 2 numeric columns")
            
            with stat_tabs[3]:
                st.markdown("### Distribution Tests")
                
                # Select columns for distribution tests
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    selected_col = st.selectbox("Select column for distribution test", numeric_cols)
                    
                    # Histogram with normal distribution overlay
                    data = df[selected_col].dropna()
                    
                    fig = make_subplots(rows=2, cols=1, 
                                       subplot_titles=("Histogram with Normal Distribution", "Q-Q Plot"),
                                       vertical_spacing=0.15,
                                       row_heights=[0.7, 0.3])
                    
                    # Histogram
                    hist_values, bin_edges = np.histogram(data, bins=30, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    fig.add_trace(
                        go.Bar(
                            x=bin_centers,
                            y=hist_values,
                            name="Histogram",
                            marker_color="lightblue"
                        ),
                        row=1, col=1
                    )
                    
                    # Normal distribution overlay
                    from scipy.stats import norm
                    mean, std = data.mean(), data.std()
                    x = np.linspace(data.min(), data.max(), 100)
                    y = norm.pdf(x, mean, std)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            name="Normal Distribution",
                            line=dict(color="red")
                        ),
                        row=1, col=1
                    )
                    
                    # Q-Q plot
                    from scipy.stats import probplot
                    result = probplot(data, dist="norm", plot=None)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=result[0][0],
                            y=result[0][1],
                            mode='markers',
                            name="Q-Q Plot",
                            marker_color="blue"
                        ),
                        row=2, col=1
                    )
                    
                    # Reference line
                    line_x = np.array([result[0][0].min(), result[0][0].max()])
                    line_y = result[1][0] + result[1][1] * line_x
                    
                    fig.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode='lines',
                            name="Reference Line",
                            line=dict(color="red", dash="dash")
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical tests
                    st.markdown('<div class="section-header">Normality Tests</div>', unsafe_allow_html=True)
                    
                    from scipy.stats import shapiro, kstest
                    
                    # Shapiro-Wilk test
                    try:
                        stat, p_val = shapiro(data)
                        
                        st.write("#### Shapiro-Wilk Test")
                        st.write(f"Statistic: {stat:.4f}")
                        st.write(f"p-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.write("Result: The data is likely NOT normally distributed (p < 0.05)")
                        else:
                            st.write("Result: The data is likely normally distributed (p >= 0.05)")
                    except Exception as e:
                        st.warning(f"Could not perform Shapiro-Wilk test: {str(e)}")
                    
                    # Kolmogorov-Smirnov test
                    try:
                        stat, p_val = kstest(data, 'norm', args=(data.mean(), data.std()))
                        
                        st.write("#### Kolmogorov-Smirnov Test")
                        st.write(f"Statistic: {stat:.4f}")
                        st.write(f"p-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.write("Result: The data is likely NOT normally distributed (p < 0.05)")
                        else:
                            st.write("Result: The data is likely normally distributed (p >= 0.05)")
                    except Exception as e:
                        st.warning(f"Could not perform Kolmogorov-Smirnov test: {str(e)}")
                else:
                    st.warning("Distribution tests require at least one numeric column")
    
    # Time Series Analysis
    elif app_mode == "Time Series Analysis":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload and Preprocessing' tab first.")
        else:
            df = st.session_state.df
            
            st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)
            
            # Check for datetime columns
            datetime_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_dtype(df[col]):
                    datetime_cols.append(col)
                else:
                    try:
                        # Try to convert to datetime
                        pd.to_datetime(df[col])
                        datetime_cols.append(col)
                    except:
                        pass
            
            if datetime_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    date_col = st.selectbox("Select date column", datetime_cols)
                
                with col2:
                    # Filter numeric columns for time series
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        value_col = st.selectbox("Select value column", numeric_cols)
                        
                        # Perform time series analysis
                        perform_time_series_analysis(df, date_col, value_col)
                    else:
                        st.warning("Time series analysis requires at least one numeric column")
            else:
                st.warning("No date/time columns detected. Please preprocess your data to identify date columns.")
                
                # Offer to convert a column to datetime
                cols_to_convert = st.selectbox("Select a column to convert to datetime", df.columns)
                
                if st.button("Convert to datetime"):
                    try:
                        df[cols_to_convert] = pd.to_datetime(df[cols_to_convert])
                        st.session_state.df = df
                        st.success(f"Successfully converted '{cols_to_convert}' to datetime!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error converting to datetime: {str(e)}")
    
    # Clustering Analysis
    elif app_mode == "Clustering Analysis":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload and Preprocessing' tab first.")
        else:
            df = st.session_state.df
            
            st.markdown('<div class="section-header">Clustering Analysis</div>', unsafe_allow_html=True)
            
            # Select features for clustering
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                selected_features = st.multiselect(
                    "Select features for clustering",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if len(selected_features) >= 1:
                    # Handle missing values for selected features
                    df_cluster = df[selected_features].copy()
                    
                    # Check for missing values
                    if df_cluster.isnull().sum().sum() > 0:
                        st.warning("Selected features contain missing values. These will be imputed with the mean.")
                        imputer = SimpleImputer(strategy='mean')
                        df_cluster = pd.DataFrame(
                            imputer.fit_transform(df_cluster),
                            columns=selected_features
                        )
                    
                    # Perform clustering analysis
                    perform_clustering(df, selected_features)
                else:
                    st.warning("Please select at least one feature for clustering")
            else:
                st.warning("Clustering analysis requires at least 2 numeric columns")
    
    # Export Data
    elif app_mode == "Export Data":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload and Preprocessing' tab first.")
        else:
            df = st.session_state.df
            
            st.markdown('<div class="section-header">Export Processed Data</div>', unsafe_allow_html=True)
            
            # Show data summary
            st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
            st.dataframe(df.head(), use_container_width=True)
            
            # Select export format
            export_format = st.radio("Select export format", ["CSV", "Excel", "JSON"])
            
            if export_format == "CSV":
                export_data(df, "csv")
            elif export_format == "Excel":
                try:
                    from io import BytesIO
                    import xlsxwriter
                    export_data(df, "excel")
                except ImportError:
                    st.error("Excel export requires the xlsxwriter package. Please install it or choose another format.")
            else:  # JSON
                export_data(df, "json")
            
            # Option to reset all data
            if st.button("Reset All Data", key="reset_all"):
                st.session_state.df = None
                st.session_state.original_df = None
                st.success("All data has been reset. You can now upload a new dataset.")
                st.experimental_rerun()

if __name__ == "__main__":
    main()

    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "json"])
    
    if uploaded_file is not None:
        file_type = st.radio("Select file type", ["CSV", "Excel", "JSON"])
        df = load_data(uploaded_file, file_type.lower())
        if df is not None:
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    file_option = st.radio("Select data source", ["Upload File", "Enter URL"])
    
    if file_option == "Enter URL":
        url = st.text_input("Enter URL to CSV/Excel/JSON file")
        file_type = st.radio("Select file type", ["CSV", "Excel", "JSON"])
        
        if url and url.strip():
            try:
                df = load_data(url, file_type.lower())
                if df is not None:
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.success(f"Successfully loaded data from URL with {df.shape[0]} rows and {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading data from URL: {str(e)}")