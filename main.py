import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
import io
import uuid
import os

# Set page configuration
st.set_page_config(
    page_title="Financial Data Analysis App",
    page_icon="💰",
    layout="wide"
)

# Apply green, white, and black themed styling
st.markdown("""
<style>
    body {
        background-color: #ffffff;
        color: #000000;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        color: #006400;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1.2rem;
        color: #006400;
    }
    .stButton>button {
        background-color: #006400;
        color: white;
    }
    .stSelectbox>div>div {
        background-color: #f0fff0;
    }
    .stDataFrame {
        border: 1px solid #006400;
    }
    .css-1d391kg {
        background-color: #f0fff0;
    }
    .sidebar .sidebar-content {
        background-color: #f0fff0;
    }
    .sidebar .sidebar-content .sidebar-section {
        background-color: #f0fff0;
    }
</style>
""", unsafe_allow_html=True)

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

def main():
    st.markdown('<div class="main-header">💰 Financial Data Analysis App</div>', unsafe_allow_html=True)
    
    # Create sidebar for navigation 
    with st.sidebar:
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        app_mode = st.radio("Choose Mode", 
                          ["Data Upload", 
                           "Column Management",
                           "Data Exploration",
                           "Visualization",
                           "Data Aggregation",
                           "Export Data"])
        
        st.markdown('<div class="section-header">Sample Data</div>', unsafe_allow_html=True)
        sample_data = st.selectbox(
            "Load sample dataset",
            ["None", "Bank Transactions", "Credit Card"]
        )
    
    # Initialize session state
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {}  # Store multiple dataframes
    
    if 'combined_df' not in st.session_state:
        st.session_state.combined_df = None  # Combined dataframe
    
    if 'column_mappings' not in st.session_state:
        st.session_state.column_mappings = {}  # For storing column name mappings
    
    # Load sample data if selected
    if sample_data != "None" and len(st.session_state.dataframes) == 0:
        if sample_data == "Bank Transactions":
            # Create sample bank transaction data
            dates = pd.date_range(start='2023-01-01', end='2023-03-31')
            
            # Bank 1 transactions
            np.random.seed(42)
            bank1_data = {
                'Date': np.random.choice(dates, 100),
                'Amount': np.random.uniform(-2000, 5000, 100).round(2),
                'Description': np.random.choice(['Salary', 'Groceries', 'Rent', 'Utilities', 'Entertainment', 'Transfer'], 100),
                'Category': np.random.choice(['Income', 'Expense', 'Transfer'], 100),
                'Account': 'Checking'
            }
            bank1_df = pd.DataFrame(bank1_data)
            
            # Bank 2 transactions
            np.random.seed(43)
            bank2_data = {
                'TransactionDate': np.random.choice(dates, 80),
                'TransactionAmount': np.random.uniform(-1500, 3000, 80).round(2),
                'Merchant': np.random.choice(['Paycheck', 'Supermarket', 'Housing', 'Bills', 'Dining', 'Transfer'], 80),
                'Type': np.random.choice(['Credit', 'Debit', 'Transfer'], 80),
                'AccountType': 'Savings'
            }
            bank2_df = pd.DataFrame(bank2_data)
            
            # Add to session state
            st.session_state.dataframes['bank1'] = bank1_df
            st.session_state.dataframes['bank2'] = bank2_df
            
            st.success(f"Loaded sample bank transaction data with 2 files")
                
        elif sample_data == "Credit Card":
            # Create sample credit card data
            dates = pd.date_range(start='2023-01-01', end='2023-06-30')
            
            # Credit card 1
            np.random.seed(44)
            cc1_data = {
                'Transaction_Date': np.random.choice(dates, 150),
                'Amount': np.random.uniform(-500, 1000, 150).round(2),
                'Merchant_Name': np.random.choice(['Restaurant', 'Gas Station', 'Online Shopping', 'Grocery Store', 'Utility', 'Subscription'], 150),
                'Category': np.random.choice(['Food', 'Transportation', 'Shopping', 'Groceries', 'Bills', 'Entertainment'], 150),
                'Card_Type': 'Visa'
            }
            cc1_df = pd.DataFrame(cc1_data)
            
            # Credit card 2
            np.random.seed(45)
            cc2_data = {
                'Date': np.random.choice(dates, 120),
                'Charge': np.random.uniform(-600, 800, 120).round(2),
                'Vendor': np.random.choice(['Dining', 'Travel', 'Retail', 'Supermarket', 'Service', 'Digital'], 120),
                'Expense_Type': np.random.choice(['Dining', 'Travel', 'Shopping', 'Groceries', 'Utilities', 'Subscription'], 120),
                'Card': 'Mastercard'
            }
            cc2_df = pd.DataFrame(cc2_data)
            
            # Add to session state
            st.session_state.dataframes['cc1'] = cc1_df
            st.session_state.dataframes['cc2'] = cc2_df
            
            st.success(f"Loaded sample credit card data with 2 files")
    
    # Data Upload Mode
    if app_mode == "Data Upload":
        st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("Upload file(s)", type=["csv", "xlsx", "json"], accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Generate a unique ID for this file if it doesn't exist
                file_id = uploaded_file.name.replace(".", "_").replace(" ", "_")
                
                # Only process if this file hasn't been loaded yet
                if file_id not in st.session_state.dataframes:
                    file_type = st.radio(f"Select file type for {uploaded_file.name}", 
                                       ["CSV", "Excel", "JSON"], key=f"file_type_{file_id}")
                    
                    if st.button(f"Load {uploaded_file.name}", key=f"load_{file_id}"):
                        df = load_data(uploaded_file, file_type.lower(), uploaded_file.name)
                        if df is not None:
                            st.session_state.dataframes[file_id] = df
                            st.success(f"Successfully loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Display loaded files
        if st.session_state.dataframes:
            st.markdown('<div class="section-header">Loaded Files</div>', unsafe_allow_html=True)
            
            for file_id, df in st.session_state.dataframes.items():
                with st.expander(f"{file_id} - {df.shape[0]} rows, {df.shape[1]} columns"):
                    st.dataframe(df.head(), use_container_width=True)
            
            # Combine files option
            st.markdown('<div class="section-header">Combine Files</div>', unsafe_allow_html=True)
            
            if st.button("Combine All Files"):
                # Combine all dataframes
                combined_df = pd.concat(st.session_state.dataframes.values(), ignore_index=True)
                st.session_state.combined_df = combined_df
                st.success(f"Successfully combined {len(st.session_state.dataframes)} files into one dataset with {combined_df.shape[0]} rows")
            
            # Data preprocessing options
            if st.session_state.combined_df is not None:
                df = st.session_state.combined_df
                
                st.markdown('<div class="section-header">Combined Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)
                
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
                if st.button("Save Preprocessing Changes"):
                    st.session_state.combined_df = df
                    st.success("Changes saved! You can now proceed to column management and analysis.")
    
    # Column Management Mode
    elif app_mode == "Column Management":
        if not st.session_state.dataframes:
            st.warning("Please upload data in the 'Data Upload' tab first.")
        elif st.session_state.combined_df is None:
            st.warning("Please combine your files in the 'Data Upload' tab first.")
        else:
            df = st.session_state.combined_df
            
            st.markdown('<div class="section-header">Column Management</div>', unsafe_allow_html=True)
            
            # Column renaming
            st.markdown('<div class="section-header">Rename Columns</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            # Initialize column mappings if empty
            if not st.session_state.column_mappings:
                for col in df.columns:
                    st.session_state.column_mappings[col] = col
            
            rename_columns_dict = {}
            for i, col in enumerate(df.columns):
                with col1:
                    if i % 2 == 0:
                        new_name = st.text_input(f"Rename '{col}'", value=st.session_state.column_mappings.get(col, col), key=f"rename_{i}")
                        st.session_state.column_mappings[col] = new_name
                        rename_columns_dict[col] = new_name
                with col2:
                    if i % 2 == 1:
                        new_name = st.text_input(f"Rename '{col}'", value=st.session_state.column_mappings.get(col, col), key=f"rename_{i}")
                        st.session_state.column_mappings[col] = new_name
                        rename_columns_dict[col] = new_name
            
            if st.button("Apply Column Renames"):
                try:
                    df = rename_columns(df, rename_columns_dict)
                    st.session_state.combined_df = df
                    st.success("Column names updated successfully!")
                except Exception as e:
                    st.error(f"Error renaming columns: {str(e)}")
            
            # Column removal
            st.markdown('<div class="section-header">Remove Columns</div>', unsafe_allow_html=True)
            
            columns_to_remove = st.multiselect(
                "Select columns to remove",
                df.columns
            )
            
            if columns_to_remove and st.button("Remove Selected Columns"):
                try:
                    df = df.drop(columns=columns_to_remove)
                    st.session_state.combined_df = df
                    st.success(f"Successfully removed {len(columns_to_remove)} columns")
                except Exception as e:
                    st.error(f"Error removing columns: {str(e)}")
            
            # Data type conversion
            st.markdown('<div class="section-header">Change Data Types</div>', unsafe_allow_html=True)
            
            with st.expander("Change column data types"):
                for col in df.columns:
                    col_type = st.selectbox(
                        f"Convert '{col}' from {df[col].dtype}",
                        ["Keep current", "string", "integer", "float", "categorical", "boolean", "datetime"],
                        key=f"convert_{col}"
                    )
                    
                    if col_type != "Keep current" and st.button(f"Convert {col}", key=f"btn_convert_{col}"):
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
                            
                            st.session_state.combined_df = df
                            st.success(f"Converted '{col}' to {col_type}")
                        except Exception as e:
                            st.error(f"Error converting '{col}' to {col_type}: {str(e)}")
            
            # Preview updated dataframe
            st.markdown('<div class="section-header">Updated Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
    
    # Data Exploration Mode
    elif app_mode == "Data Exploration":
        if not st.session_state.dataframes:
            st.warning("Please upload data in the 'Data Upload' tab first.")
        elif st.session_state.combined_df is None:
            st.warning("Please combine your files in the 'Data Upload' tab first.")
        else:
            df = st.session_state.combined_df
            
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
            
            # Source file statistics (if available)
            if '_source_file' in df.columns:
                st.markdown('<div class="section-header">Source File Statistics</div>', unsafe_allow_html=True)
                source_counts = df['_source_file'].value_counts().reset_index()
                source_counts.columns = ['Source File', 'Count']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(source_counts, use_container_width=True)
                
                with col2:
                    fig = px.pie(source_counts, values='Count', names='Source File',
                               title="Data Distribution by Source File",
                               color_discrete_sequence=px.colors.sequential.Greens)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Univariate analysis
            st.markdown('<div class="section-header">Univariate Analysis</div>', unsafe_allow_html=True)
            selected_column = st.selectbox("Select column to analyze", df.columns)
            perform_univariate_analysis(df, selected_column)
    
    # Visualization Mode
    elif app_mode == "Visualization":
        if not st.session_state.dataframes:
            st.warning("Please upload data in the 'Data Upload' tab first.")
        elif st.session_state.combined_df is None:
            st.warning("Please combine your files in the 'Data Upload' tab first.")
        else:
            df = st.session_state.combined_df
            
            st.markdown('<div class="section-header">Data Visualization</div>', unsafe_allow_html=True)
            
            viz_type = st.selectbox(
                "Select visualization type", 
                ["Bivariate Analysis", "Correlation Heatmap", "Scatter Matrix", "Time Series (if applicable)"]
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
                            color_continuous_scale="Viridis", # Using green scale
                            zmin=-1, zmax=1,
                            title="Correlation Matrix"
                        )
                        fig.update_layout(height=600, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
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
                    
                    color_by = st.selectbox("Color by (optional)", 
                                          ["None"] + df.columns.tolist())
                    
                    if len(selected_cols) >= 2:
                        # Create scatter matrix
                        if color_by != "None":
                            fig = px.scatter_matrix(
                                df, 
                                dimensions=selected_cols,
                                color=color_by,
                                title="Scatter Matrix",
                                color_discrete_sequence=px.colors.sequential.Greens if pd.api.types.is_numeric_dtype(df[color_by]) else px.colors.qualitative.Set3
                            )
                        else:
                            fig = px.scatter_matrix(
                                df, 
                                dimensions=selected_cols,
                                title="Scatter Matrix",
                                color_discrete_sequence=['#006400']  # Green color
                            )
                        
                        fig.update_layout(height=800, plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
                        fig.update_traces(diagonal_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 columns for scatter matrix")
                else:
                    st.warning("Scatter matrix requires at least 2 numeric columns")
            
            elif viz_type == "Time Series (if applicable)":
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
                    date_col = st.selectbox("Select date column", datetime_cols)
                    
                    # Select numeric column for time series
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numeric_cols:
                        value_col = st.selectbox("Select value column", numeric_cols)
                        
                        # Convert to datetime if not already
                        if not pd.api.types.is_datetime64_dtype(df[date_col]):
                            try:
                                df[date_col] = pd.to_datetime(df[date_col])
                            except Exception as e:
                                st.error(f"Could not convert {date_col} to datetime: {str(e)}")
                                st.stop()
                        
                        # Create time series plot
                        fig = px.line(df.sort_values(by=date_col), x=date_col, y=value_col,
                                    title=f"{value_col} over time", color_discrete_sequence=['#006400'])
                        
                        # Add optional grouping
                        group_by = st.selectbox("Group by (optional)", 
                                              ["None"] + [col for col in df.columns if col != date_col and col != value_col])
                        
                        if group_by != "None":
                            fig = px.line(df.sort_values(by=date_col), x=date_col, y=value_col, 
                                        color=group_by, title=f"{value_col} over time grouped by {group_by}",
                                        color_discrete_sequence=px.colors.sequential.Greens_r)
                        
                        fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Aggregation period for time series
                        st.markdown('<div class="section-header">Time Period Aggregation</div>', unsafe_allow_html=True)
                        
                        agg_period = st.selectbox(
                            "Aggregate by time period", 
                            ["Day", "Week", "Month", "Quarter", "Year"]
                        )
                        
                        agg_method = st.selectbox(
                            "Aggregation method", 
                            ["Sum", "Mean", "Min", "Max", "Count"]
                        )
                        
                        if st.button("Apply Time Aggregation"):
                            # Set the date column as index
                            df_agg = df.set_index(date_col)
                            
                            # Get period code
                            period_map = {
                                "Day": "D",
                                "Week": "W",
                                "Month": "M",
                                "Quarter": "Q",
                                "Year": "Y"
                            }
                            
                            # Perform aggregation
                            if agg_method == "Sum":
                                if group_by != "None":
                                    df_agg = df_agg.groupby([pd.Grouper(freq=period_map[agg_period]), group_by])[value_col].sum().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col, color=group_by,
                                                title=f"{value_col} ({agg_method} by {agg_period}) grouped by {group_by}",
                                                color_discrete_sequence=px.colors.sequential.Greens_r)
                                else:
                                    df_agg = df_agg.resample(period_map[agg_period])[value_col].sum().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col,
                                                title=f"{value_col} ({agg_method} by {agg_period})",
                                                color_discrete_sequence=['#006400'])
                                
                            elif agg_method == "Mean":
                                if group_by != "None":
                                    df_agg = df_agg.groupby([pd.Grouper(freq=period_map[agg_period]), group_by])[value_col].mean().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col, color=group_by,
                                                title=f"{value_col} ({agg_method} by {agg_period}) grouped by {group_by}",
                                                color_discrete_sequence=px.colors.sequential.Greens_r)
                                else:
                                    df_agg = df_agg.resample(period_map[agg_period])[value_col].mean().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col,
                                                title=f"{value_col} ({agg_method} by {agg_period})",
                                                color_discrete_sequence=['#006400'])
                            
                            elif agg_method == "Min":
                                if group_by != "None":
                                    df_agg = df_agg.groupby([pd.Grouper(freq=period_map[agg_period]), group_by])[value_col].min().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col, color=group_by,
                                                title=f"{value_col} ({agg_method} by {agg_period}) grouped by {group_by}",
                                                color_discrete_sequence=px.colors.sequential.Greens_r)
                                else:
                                    df_agg = df_agg.resample(period_map[agg_period])[value_col].min().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col,
                                                title=f"{value_col} ({agg_method} by {agg_period})",
                                                color_discrete_sequence=['#006400'])
                            
                            elif agg_method == "Max":
                                if group_by != "None":
                                    df_agg = df_agg.groupby([pd.Grouper(freq=period_map[agg_period]), group_by])[value_col].max().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col, color=group_by,
                                                title=f"{value_col} ({agg_method} by {agg_period}) grouped by {group_by}",
                                                color_discrete_sequence=px.colors.sequential.Greens_r)
                                else:
                                    df_agg = df_agg.resample(period_map[agg_period])[value_col].max().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col,
                                                title=f"{value_col} ({agg_method} by {agg_period})",
                                                color_discrete_sequence=['#006400'])
                            
                            elif agg_method == "Count":
                                if group_by != "None":
                                    df_agg = df_agg.groupby([pd.Grouper(freq=period_map[agg_period]), group_by])[value_col].count().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col, color=group_by,
                                                title=f"{value_col} ({agg_method} by {agg_period}) grouped by {group_by}",
                                                color_discrete_sequence=px.colors.sequential.Greens_r)
                                else:
                                    df_agg = df_agg.resample(period_map[agg_period])[value_col].count().reset_index()
                                    fig = px.line(df_agg, x=date_col, y=value_col,
                                                title=f"{value_col} ({agg_method} by {agg_period})",
                                                color_discrete_sequence=['#006400'])
                            
                            fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show aggregated data
                            st.markdown('<div class="section-header">Aggregated Data</div>', unsafe_allow_html=True)
                            st.dataframe(df_agg, use_container_width=True)
                    else:
                        st.warning("Time series visualization requires at least one numeric column")
                else:
                    st.warning("No date/time columns detected. Please convert a column to datetime in Column Management.")
                    
                    # Data Aggregation Mode
    elif app_mode == "Data Aggregation":
        if not st.session_state.dataframes:
            st.warning("Please upload data in the 'Data Upload' tab first.")
        elif st.session_state.combined_df is None:
            st.warning("Please combine your files in the 'Data Upload' tab first.")
        else:
            df = st.session_state.combined_df
            
            st.markdown('<div class="section-header">Data Aggregation</div>', unsafe_allow_html=True)
            st.write("Aggregate your data based on selected columns and functions")
            
            # Select columns to group by
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            group_by_cols = st.multiselect(
                "Select columns to group by",
                categorical_cols
            )
            
            # Select columns to aggregate
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_cols = st.multiselect(
                "Select columns to aggregate",
                numeric_cols
            )
            
            # Select aggregation functions
            agg_functions = st.multiselect(
                "Select aggregation functions",
                ["sum", "mean", "median", "min", "max", "count", "std", "var"]
            )
            
            if group_by_cols and agg_cols and agg_functions and st.button("Perform Aggregation"):
                aggregated_df = perform_data_aggregation(df, group_by_cols, agg_cols, agg_functions)
                
                if aggregated_df is not None:
                    st.markdown('<div class="section-header">Aggregation Results</div>', unsafe_allow_html=True)
                    st.dataframe(aggregated_df, use_container_width=True)
                    
                    # Visualization of aggregated data
                    st.markdown('<div class="section-header">Visualization</div>', unsafe_allow_html=True)
                    
                    # If we have only one group by column, we can create bar charts
                    if len(group_by_cols) == 1:
                        for col in aggregated_df.columns:
                            if col != group_by_cols[0] and pd.api.types.is_numeric_dtype(aggregated_df[col]):
                                fig = px.bar(
                                    aggregated_df, 
                                    x=group_by_cols[0], 
                                    y=col,
                                    title=f"{col} by {group_by_cols[0]}",
                                    color=col,
                                    color_continuous_scale="Viridis"  # Green color scale
                                )
                                fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # If we have two group by columns, we can create a heatmap
                    elif len(group_by_cols) == 2:
                        # Only for the first aggregated column
                        if len(aggregated_df.columns) > 2:
                            agg_col = [col for col in aggregated_df.columns if col not in group_by_cols][0]
                            
                            # Create pivot table
                            pivot_df = aggregated_df.pivot(
                                index=group_by_cols[0],
                                columns=group_by_cols[1],
                                values=agg_col
                            )
                            
                            fig = px.imshow(
                                pivot_df,
                                title=f"Heatmap of {agg_col} by {group_by_cols[0]} and {group_by_cols[1]}",
                                color_continuous_scale="Viridis"  # Using green scale
                            )
                            fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Export aggregated data
                    csv = aggregated_df.to_csv(index=False)
                    st.download_button(
                        label="Download Aggregated Data as CSV",
                        data=csv,
                        file_name="aggregated_data.csv",
                        mime="text/csv"
                    )
    
    # Export Data Mode
    elif app_mode == "Export Data":
        if not st.session_state.dataframes:
            st.warning("Please upload data in the 'Data Upload' tab first.")
        elif st.session_state.combined_df is None:
            st.warning("Please combine your files in the 'Data Upload' tab first.")
        else:
            df = st.session_state.combined_df
            
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
                st.session_state.dataframes = {}
                st.session_state.combined_df = None
                st.session_state.column_mappings = {}
                st.success("All data has been reset. You can now upload new files.")
                st.experimental_rerun()

if __name__ == "__main__":
    main()