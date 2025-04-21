import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
import io

# Helper functions
def load_data(file_path, file_type="csv", file_name=None):
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
        
        # Add source column to identify the file
        if file_name:
            df['_source_file'] = file_name
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def clean_column_names(df):
    """Clean column names by replacing spaces with underscores and converting to lowercase"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df

def rename_columns(df, name_mapping):
    """Rename columns based on mapping"""
    return df.rename(columns=name_mapping)

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
        
        fig = px.bar(memory_usage, y='Column', x='Memory (MB)', orientation='h',
                   color='Memory (MB)', color_continuous_scale='Viridis')  # Using green color scale
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
                go.Histogram(x=df[column], name='Histogram', marker_color='#006400'), # Green color
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(x=df[column], name='Box Plot', marker_color='#006400'), # Green color
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=False, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Bar chart for categorical
            value_counts = df[column].value_counts().reset_index().head(20)
            value_counts.columns = [column, 'Count']
            
            fig = px.bar(value_counts, x=column, y='Count',
                       title=f"Top 20 values for {column}", color='Count',
                       color_continuous_scale='Viridis') # Green color scale
            fig.update_layout(height=500, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
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
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}",
                          color_discrete_sequence=['#006400']) # Green color
            fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate correlation
            correlation = df[[x_col, y_col]].corr().iloc[0, 1]
            st.metric("Correlation coefficient", f"{correlation:.4f}")
            
            # Scatter plot with trend line
            fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                          title=f"{x_col} vs {y_col} with trend line",
                          color_discrete_sequence=['#006400']) # Green color
            fig.update_traces(line=dict(color='#000000')) # Black trendline
            fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
            st.plotly_chart(fig, use_container_width=True)
            
    elif x_is_numeric and not y_is_numeric:
        # Box plot for numeric vs categorical
        fig = px.box(df, x=y_col, y=x_col, title=f"Distribution of {x_col} by {y_col}",
                   color=y_col, color_discrete_sequence=px.colors.sequential.Greens)
        fig.update_layout(height=500, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
        st.plotly_chart(fig, use_container_width=True)
        
    elif not x_is_numeric and y_is_numeric:
        # Box plot for categorical vs numeric
        fig = px.box(df, x=x_col, y=y_col, title=f"Distribution of {y_col} by {x_col}",
                   color=x_col, color_discrete_sequence=px.colors.sequential.Greens)
        fig.update_layout(height=500, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Heatmap for two categorical columns
        contingency_table = pd.crosstab(df[x_col], df[y_col])
        fig = px.imshow(contingency_table, text_auto=True,
                       title=f"Contingency table: {x_col} vs {y_col}",
                       color_continuous_scale='Viridis') # Green color scale
        fig.update_layout(height=600, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
        st.plotly_chart(fig, use_container_width=True)

def perform_data_aggregation(df, group_by_cols, agg_cols, agg_functions):
    """
    Perform data aggregation based on user selections
    
    Args:
        df: DataFrame to aggregate
        group_by_cols: Columns to group by
        agg_cols: Columns to aggregate
        agg_functions: Aggregation functions to apply
        
    Returns:
        Aggregated DataFrame
    """
    if not group_by_cols or not agg_cols or not agg_functions:
        st.warning("Please select at least one column to group by, one column to aggregate, and one aggregation function")
        return None
    
    # Create aggregation dictionary
    agg_dict = {}
    for col in agg_cols:
        agg_dict[col] = agg_functions
    
    # Perform aggregation
    try:
        result = df.groupby(group_by_cols).agg(agg_dict)
        
        # Flatten multi-level column names if needed
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
        
        # Reset index for better display
        result = result.reset_index()
        return result
    except Exception as e:
        st.error(f"Error during aggregation: {str(e)}")
        return None

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