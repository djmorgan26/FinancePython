import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import io

# Set page configuration
st.set_page_config(
    page_title="Simple Data Analysis App",
    page_icon="📊",
    layout="wide"
)

# Apply some basic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache mechanism for optimizing data loading
@st.cache_data(ttl=3600)
def load_data(file_path, file_type="csv"):
    """Load data from a file and return it as a pandas DataFrame."""
    try:
        if file_type == "csv":
            # Try to detect separator
            if isinstance(file_path, str):
                with open(file_path, 'r') as f:
                    sample = f.read(5000)
            else:
                sample = file_path.getvalue()[:5000].decode('utf-8')
            
            if ',' in sample:
                sep = ','
            elif ';' in sample:
                sep = ';'
            elif '\t' in sample:
                sep = '\t'
            else:
                sep = None
                
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
    else:
        return df

def display_dashboard(df):
    """Display basic dashboard for the dataset"""
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    col4.metric("Duplicated Rows", f"{df.duplicated().sum():,}")
    
    # Data types
    dtypes = df.dtypes.value_counts().reset_index()
    dtypes.columns = ['Data Type', 'Count']
    
    col1, col2 = st.columns(2)
    
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
        
        fig = px.bar(memory_usage, y='Column', x='Memory (MB)', orientation='h')
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

def perform_clustering(df, selected_features, n_clusters=3):
    """Perform K-means clustering on selected features"""
    st.markdown('<div class="section-header">K-means Clustering</div>', unsafe_allow_html=True)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[selected_features])
    
    # Perform clustering
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Perform PCA for visualization if we have more than 2 features
    if len(selected_features) > 2:
        # Use PCA to visualize in 2D
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
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
        output = io.BytesIO()
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
    st.markdown('<div class="main-header">📊 Simple Data Analysis App</div>', unsafe_allow_html=True)
    
    # Create sidebar for navigation 
    with st.sidebar:
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        app_mode = st.radio("Choose Mode", 
                          ["Data Upload", 
                           "Data Exploration",
                           "Visualization",
                           "Clustering",
                           "Export Data"])
        
        st.markdown('<div class="section-header">Sample Data</div>', unsafe_allow_html=True)
        sample_data = st.selectbox(
            "Load sample dataset",
            ["None", "Iris", "Titanic"]
        )
    
    # Initialize session state for DataFrame
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.original_df = None
    
    # Load sample data if selected
    if sample_data != "None":
        if sample_data == "Iris":
            try:
                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(iris.data, columns=iris.feature_names)
                df['species'] = iris.target
                df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            except:
                # Fallback if sklearn dataset unavailable
                url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
                df = pd.read_csv(url)
                
        elif sample_data == "Titanic":
            try:
                url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                df = pd.read_csv(url)
            except:
                st.error("Could not load Titanic dataset")
                df = None
        
        if df is not None:
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.success(f"Loaded {sample_data} dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Data Upload Mode
    if app_mode == "Data Upload":
        st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "json"])
        
        if uploaded_file is not None:
            file_type = st.radio("Select file type", ["CSV", "Excel", "JSON"])
            
            if st.button("Load Data"):
                df = load_data(uploaded_file, file_type.lower())
                if df is not None:
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Display data preview if available
        if st.session_state.df is not None:
            df = st.session_state.df
            
            st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # Data preprocessing options
            st.markdown('<div class="section-header">Data Preprocessing</div>', unsafe_allow_html=True)
            
            preprocessing_options = st.multiselect(
                "Select preprocessing steps",
                ["Clean Column Names", "Drop Duplicates", "Handle Missing Values"]
            )
            
            if "Clean Column Names" in preprocessing_options:
                df = clean_column_names(df)
                st.success("Column names cleaned!")
                
            if "Drop Duplicates" in preprocessing_options:
                original_rows = df.shape[0]
                df = df.drop_duplicates()
                new_rows = df.shape[0]
                st.write(f"Removed {original_rows - new_rows} duplicate rows.")
            
            if "Handle Missing Values" in preprocessing_options:
                missing_df = check_missing_values(df)
                if missing_df.empty:
                    st.success("No missing values found in the dataset!")
                else:
                    st.write("Missing values found:")
                    st.dataframe(missing_df)
                    
                    missing_method = st.selectbox(
                        "Select method to handle missing values",
                        ["Do nothing", "Drop rows", "Fill with mean", "Fill with median", "Fill with zero"]
                    )
                    
                    if missing_method != "Do nothing" and st.button("Apply"):
                        if missing_method == "Drop rows":
                            df = handle_missing_values(df, "drop")
                        elif missing_method == "Fill with mean":
                            df = handle_missing_values(df, "mean")
                        elif missing_method == "Fill with median":
                            df = handle_missing_values(df, "median")
                        elif missing_method == "Fill with zero":
                            df = handle_missing_values(df, "zero")
                        
                        st.success(f"Missing values handled using '{missing_method}'!")
            
            # Save preprocessed data
            if st.button("Save Changes"):
                st.session_state.df = df
                st.success("Changes saved! You can now proceed to analysis.")
    
    # Data Exploration Mode
    elif app_mode == "Data Exploration":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload' tab first.")
        else:
            df = st.session_state.df
            
            # Display the dashboard overview
            display_dashboard(df)
            
            # Column details
            st.markdown('<div class="section-header">Column Details</div>', unsafe_allow_html=True)
            
            # Get column statistics
            col_stats = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            
            st.dataframe(col_stats, use_container_width=True)
            
            # Univariate analysis
            st.markdown('<div class="section-header">Univariate Analysis</div>', unsafe_allow_html=True)
            selected_column = st.selectbox("Select column to analyze", df.columns)
            perform_univariate_analysis(df, selected_column)
    
    # Visualization Mode
    elif app_mode == "Visualization":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload' tab first.")
        else:
            df = st.session_state.df
            
            st.markdown('<div class="section-header">Data Visualization</div>', unsafe_allow_html=True)
            
            viz_type = st.selectbox(
                "Select visualization type", 
                ["Bivariate Analysis", "Correlation Heatmap", "Scatter Matrix"]
            )
            
            if viz_type == "Bivariate Analysis":
                col1, col2 = st.columns(2)
                
                with col1:
                    x_column = st.selectbox("Select X-axis column", df.columns, key="x_col")
                
                with col2:
                    # Filter out the already selected column
                    available_cols = [col for col in df.columns if col != x_column]
                    y_column = st.selectbox("Select Y-axis column", available_cols, key="y_col")
                
                # Perform bivariate analysis
                perform_bivariate_analysis(df, x_column, y_column)
            
            elif viz_type == "Correlation Heatmap":
                # Select numeric columns for correlation
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    selected_cols = st.multiselect(
                        "Select columns for correlation",
                        numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))]
                    )
                    
                    if len(selected_cols) >= 2:
                        # Calculate correlation matrix
                        corr = df[selected_cols].corr()
                        
                        # Display correlation matrix as heatmap
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title="Correlation Matrix"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 columns for correlation")
                else:
                    st.warning("Correlation heatmap requires at least 2 numeric columns")
            
            elif viz_type == "Scatter Matrix":
                # Select numeric columns for scatter matrix
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    selected_cols = st.multiselect(
                        "Select columns for scatter matrix",
                        numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))]
                    )
                    
                    if len(selected_cols) >= 2:
                        # Create scatter matrix
                        fig = px.scatter_matrix(
                            df, 
                            dimensions=selected_cols,
                            title="Scatter Matrix"
                        )
                        fig.update_layout(height=800)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 columns for scatter matrix")
                else:
                    st.warning("Scatter matrix requires at least 2 numeric columns")
    
    # Clustering Mode
    elif app_mode == "Clustering":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload' tab first.")
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
                    temp_df = df.copy()  # Create a copy to avoid modifying the original
                    perform_clustering(temp_df, selected_features)
                else:
                    st.warning("Please select at least one feature for clustering")
            else:
                st.warning("Clustering analysis requires at least 2 numeric columns")
    
    # Export Data Mode
    elif app_mode == "Export Data":
        if st.session_state.df is None:
            st.warning("Please upload data in the 'Data Upload' tab first.")
        else:
            df = st.session_state.df
            
            st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
            
            # Show data summary
            st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
            st.dataframe(df.head(), use_container_width=True)
            
            # Select export format
            export_format = st.radio("Select export format", ["CSV", "Excel", "JSON"])
            
            if export_format == "CSV":
                export_data(df, "csv")
            elif export_format == "Excel":
                try:
                    export_data(df, "excel")
                except ImportError:
                    st.error("Excel export requires the xlsxwriter package. Please install it or choose another format.")
            else:  # JSON
                export_data(df, "json")
            
            # Option to reset all data
            if st.button("Reset All Data"):
                st.session_state.df = None
                st.session_state.original_df = None
                st.success("All data has been reset. You can now upload a new dataset.")
                st.experimental_rerun()

if __name__ == "__main__":
    main()