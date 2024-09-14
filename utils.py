import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Misc
def convert_strings_to_booleans(string_list):
    """
    Converts a list of strings representing boolean values to their respective Python boolean values.

    Parameters
    ----------
    string_list : list of str
        A list of strings where each string is either 'true' or 'false'.

    Returns
    -------
    list of bool
        A list of boolean values corresponding to the input strings.
    """
    # Convert strings to lowercase and map 'true' to True and 'false' to False
    boolean_list = [s.lower() == 'true' for s in string_list]
    return boolean_list

# Visualization

def plot_grid_outage(outage_plan, start=None, end=None):
    """
    Plots the grid outage plan (a list of boolean values) within a specified range.

    Parameters
    ----------
    outage_plan : list of bool
        A list of boolean values representing the grid outage plan (True for outage, False for no outage).
    start : int, optional
        The starting index for the plot (default is the beginning of the list).
    end : int, optional
        The ending index for the plot (default is the end of the list).

    Returns
    -------
    None
    """
    # Handle start and end range, ensuring they are within bounds
    if start is None:
        start = 0
    if end is None:
        end = len(outage_plan)
    
    # Slice the list based on the provided range
    outage_slice = outage_plan[start:end]
    
    # Convert boolean values to integers (True -> 1, False -> 0)
    outage_as_int = [int(b) for b in outage_slice]
    
    # Create a range for the x-axis
    x_values = range(start, end)
    
    # Plotting the outage plan
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, outage_as_int, label='Grid Outage (True = Outage)', color='C0')
    
    # Add labels and title
    plt.xlabel('Index (Time)')
    plt.ylabel('Grid Outage (0 = No Outage, 1 = Outage)')
    plt.title('Grid Outage Plan')
    plt.legend()
    
    # Display the plot
    plt.show()

def plot_time_series_site_variables(data, site_name, variables, start=None, end=None, figsize=(22, 5), legend=True):
    """
    Plots time series for a specific site and multiple variables in the data, with options 
    to limit the time range by specifying a start and end.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the time series data. Must have 'site_name', 'day', and 'hour' columns.
    site_name : str
        The name of the site to filter the data by.
    variables : list of str
        A list of column names representing the variables to be plotted.
    start : tuple, optional
        A tuple of (day, hour) representing the start of the time range to plot. 
        If None, plotting starts from the beginning of the dataset.
    end : tuple, optional
        A tuple of (day, hour) representing the end of the time range to plot. 
        If None, plotting continues until the end of the dataset.
    figsize : tuple, optional
        Size of the figure in inches (width, height). Default is (22, 5).
    legend : bool, optional
        If True, display the legend on the plot. Default is True.

    Returns
    -------
    None
        Displays the time series plot for the given site and variables within the specified range.
    
    Examples
    --------
    Plot the time series for 'ghi' and 'dni' variables for 'site_A':
    
    >>> plot_time_series_site_variables(data, 'site_A', ['ghi', 'dni'])
    
    Plot only the data from day 1, hour 0 to day 3, hour 23 for 'ghi' and 'dni':
    
    >>> plot_time_series_site_variables(data, 'site_A', ['ghi', 'dni'], start=(1, 0), end=(3, 23))
    """
    # Filter data for the specified site
    filtered_data = data.query(f"site_name == '{site_name}'").set_index(['day', 'hour'])
    
    # Apply start and end filters if provided
    if start is not None:
        filtered_data = filtered_data.loc[start:]
    if end is not None:
        filtered_data = filtered_data.loc[:end]
    
    # Plot each variable
    ax = filtered_data[variables].plot(figsize=figsize, title=f"Time series for {site_name}", xlabel='Time')

    # Display or hide the legend based on the parameter
    if legend:
        ax.legend(loc='upper right')
    else:
        ax.get_legend().remove()
    
    # Display the plot
    plt.show()

def plot_hourly_energy_variation(df, figsize=(12, 6)):
    """
    Plots the variation of energy_outputkwh per hour across multiple days using a boxplot.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing 'day', 'hour', and 'energy_outputkwh'.
    figsize : tuple, optional
        The size of the figure for the plot. Default is (12, 6).

    Returns
    -------
    None
        Displays the box plot showing the variation of energy output per hour.
    """
    # Set up the figure for plotting
    plt.figure(figsize=figsize)

    # Create a boxplot to show the distribution of energy output for each hour
    sns.boxplot(x='hour', y='energy_outputkwh', data=df)

    # Set plot labels and title
    plt.xlabel('Hour of the Day')
    plt.ylabel('Energy Output (kWh)')
    plt.title('Variation of Energy Output per Hour Across 60 Days')
    plt.grid(True)
    plt.show()

def plot_means(df_output, df_total, figsize=(10, 6)):
    """
    Plots the means of energy output (energy_outputkwh) and total energy (total_energykwh)
    for each hour across all days.

    Parameters
    ----------
    df_output : pandas.DataFrame
        The DataFrame containing 'day', 'hour', and 'energy_outputkwh'.
    df_total : pandas.DataFrame
        The DataFrame containing 'day', 'hour', and 'total_energykwh'.
    figsize : tuple, optional
        The size of the figure for the plot. Default is (10, 6).

    Returns
    -------
    None
        Displays the line plot showing the means for each hour.
    """
    # Merge the two dataframes on 'day' and 'hour'
    df_merged = pd.merge(df_output, df_total, on=['day', 'hour'], how='inner')

    # Calculate mean for each hour across all days
    mean_stats = df_merged.groupby('hour')[['energy_outputkwh', 'total_energykwh']].mean().reset_index()

    # Set up the figure for plotting
    plt.figure(figsize=figsize)

    # Plot mean values across all days
    plt.plot(mean_stats['hour'], mean_stats['energy_outputkwh'], color='blue', marker='o', linestyle='-', label='Mean Energy Output', linewidth=2)
    plt.plot(mean_stats['hour'], mean_stats['total_energykwh'], color='red', marker='x', linestyle='--', label='Mean Total Energy', linewidth=2)

    # Set plot labels and title
    plt.xlabel('Hour')
    plt.ylabel('Energy (kWh)')
    plt.title('Means of Energy Output and Total Energy Across Hours')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_correlation_and_hour_variation(df, target_variable='energy_outputkwh', figsize=(14, 6)):
    """
    Computes and plots a subplot with the correlation of each variable with the target variable
    and the variation of the target variable across hours.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to analyze.
    target_variable : str, optional
        The name of the target variable to correlate with. Default is 'energy_outputkwh'.
    figsize : tuple, optional
        The size of the figure for the subplot. Default is (14, 6).

    Returns
    -------
    None
        Displays a subplot with the correlation bar plot and energy output variation across hours.
    """
    # Select relevant columns, excluding 'hour' but including the target variable
    relevant_columns = ['solar_zenith_angle', 'clearsky_dhi', 'clearsky_dni', 
                        'clearsky_ghi', 'relative_humidity', 'dhi', 'dni', 'ghi', target_variable]
    
    # Filter the DataFrame to only include relevant columns
    df_filtered = df[relevant_columns]

    # Compute the correlation of each variable with the target variable
    correlations = df_filtered.corr()[target_variable].drop(target_variable)
    
    # Sort correlations by absolute value for better visualization
    correlations_sorted = correlations.abs().sort_values(ascending=False)

    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot the bar plot of correlations
    sns.barplot(x=correlations_sorted.index, y=correlations_sorted.values, palette='coolwarm', ax=axes[0])
    axes[0].set_title(f'Correlation of Variables with {target_variable}')
    axes[0].set_ylabel('Correlation Coefficient')
    axes[0].set_xticklabels(correlations_sorted.index, rotation=45)
    axes[0].grid(True)

    # Plot the variation of energy output across hours
    sns.lineplot(x='hour', y=target_variable, data=df, ax=axes[1], marker='o')
    axes[1].set_title(f'Variation of {target_variable} Across Hours')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel(target_variable)
    axes[1].grid(True)

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, variables=None, figsize=(12, 8)):
    """
    Plots a boxplot for the specified numeric variables in a DataFrame to visually compare their ranges.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    variables : list, optional
        A list of column names to plot. If None, plots all numeric columns. Default is None.
    figsize : tuple, optional
        The size of the figure for the boxplot. Default is (12, 8).

    Returns
    -------
    None
        Displays a boxplot for the specified numeric variables in the DataFrame.
    """
    # Select only numeric columns if variables are not specified
    if variables is None:
        variables = df.select_dtypes(include='number').columns
    else:
        # Ensure the specified variables are numeric
        variables = [var for var in variables if var in df.select_dtypes(include='number').columns]
    
    # Check if there are any variables to plot
    if not variables:
        print("No numeric variables to plot.")
        return

    # Set figure size
    plt.figure(figsize=figsize)

    # Plot the boxplot for the specified variables
    sns.boxplot(data=df[variables])
    
    # Set plot details
    plt.title('Boxplot of Selected Numeric Variables')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def plot_site_variables(site_name: str, site_data, days: list = None, n_cols: int = 2, figsize: tuple = (10, 6), show_legend: bool = True):
    """
    Plots all site variables for a specific site on an hourly basis for a specified list of days. 
    If multiple days are provided, plots are arranged in a grid with a variable number of columns.

    Parameters
    ----------
    site_name : str
        The name of the site to plot data for.
    site_data : pandas.DataFrame
        A DataFrame containing the data for the specified site. It should contain columns:
        ['site_name', 'day', 'hour', 'solar_zenith_angle', 'clearsky_dhi', 'clearsky_dni',
        'clearsky_ghi', 'relative_humidity', 'dhi', 'dni', 'ghi', 'energy_outputkwh'].
    days : list, optional
        A list of days to plot. If None, plots for all days available in the site data.
    n_cols : int, optional
        The number of columns in the plot grid. Default is 2.
    figsize : tuple, optional
        The size of the figure for the plots. Default is (10, 6).
    show_legend : bool, optional
        Whether to display the legend on the plots. Default is True.

    Returns
    -------
    None
        Displays plots of all the variables for the specified days on an hourly basis.
    """
    # Filter data by the specified site name
    site_data_filtered = site_data[site_data['site_name'] == site_name]

    # If no specific days are provided, use all unique days in the data
    if days is None:
        days = site_data_filtered['day'].unique()
    
    # Get the list of variables to plot (excluding 'site_name', 'day', and 'hour')
    variables = [col for col in site_data_filtered.columns if col not in ['site_name', 'day', 'hour']]
    
    # Calculate the number of rows needed in the grid
    n_rows = (len(days) + n_cols - 1) // n_cols  # Ceiling division to get the number of rows

    # Adjust figure size and create subplots
    if n_cols == 1:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows), sharex=True, sharey=False)
        axes = axes if isinstance(axes, np.ndarray) else [axes]  # Ensure axes is always a list
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=False)
        axes = axes.flatten()  # Flatten the axes array to easily index

    # Loop through each day and create a plot
    for i, day in enumerate(days):
        # Filter data for the current day
        daily_data = site_data_filtered[site_data_filtered['day'] == day]
        
        # Plot all variables in one chart for the current day
        for var in variables:
            sns.lineplot(data=daily_data, x='hour', y=var, ax=axes[i], label=var if show_legend else None)

        # Set plot details
        axes[i].set_title(f'Site: {site_name} - Day: {day}', fontsize=12)
        axes[i].set_xlabel('Hour')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
        
        # Show or hide the legend based on the parameter
        if show_legend:
            axes[i].legend(title='Variables')
        else:
            axes[i].get_legend().remove()

    # Hide any unused subplots if days are less than n_rows * n_cols
    if n_cols > 1:
        for j in range(len(days), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def print_folder_tree(directory_path: Path, prefix: str = ""):
    """
    Recursively prints the directory tree structure starting from the given directory path.

    Parameters
    ----------
    directory_path : pathlib.Path
        The directory path to print the tree structure for.
    prefix : str, optional
        A string used to format the output, providing indentation for the tree structure.
        Default is an empty string.

    Returns
    -------
    None
        This function prints the directory tree structure to the console.
    """
    # List all items in the given directory
    items = list(directory_path.iterdir())

    # Iterate over each item
    for index, item in enumerate(items):
        # Print the item with a tree structure
        connector = "└── " if index == len(items) - 1 else "├── "
        print(prefix + connector + item.name)

        # If the item is a directory, recursively print its contents
        if item.is_dir():
            # Use the appropriate prefix for sub-items
            extension = "    " if index == len(items) - 1 else "│   "
            print_folder_tree(item, prefix + extension)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the column names of a DataFrame by converting to lowercase, removing special characters, 
    and replacing spaces with underscores.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose column names need to be cleaned.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with cleaned column names.
    """
    # Function to clean a single column name
    def clean_name(name: str) -> str:
        # Convert to lowercase
        name = name.lower()
        # Replace special characters with an empty string
        name = re.sub(r'[^a-z0-9\s_]', '', name)
        # Replace spaces with underscores
        name = re.sub(r'\s+', '_', name)
        return name

    # Apply the cleaning function to each column name
    df.columns = [clean_name(col) for col in df.columns]

    return df

def plot_histograms(df: pd.DataFrame, n_cols: int = 3, skip_columns: list = None):
    """
    Plots a grid of histograms for each numeric column in a DataFrame using Seaborn,
    skipping columns that have the same value (zero variance) or are specified in the skip list.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot histograms for.
    n_cols : int, optional
        The number of columns in the grid layout (default is 3).
    skip_columns : list, optional
        A list of column names to be excluded from plotting (default is None).

    Returns
    -------
    None
        This function displays the histogram plots for each numeric column.
    """
    if skip_columns is None:
        skip_columns = []

    # Select only numeric columns and exclude any in the skip list
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in skip_columns]

    # Filter out columns with zero variance (i.e., same value)
    numeric_columns = [col for col in numeric_columns if df[col].nunique() > 1]

    # If no columns remain after filtering, exit the function
    if not numeric_columns:
        print("No numeric columns with varying values to plot.")
        return

    # Set the style for seaborn plots
    sns.set(style="whitegrid")

    # Determine the number of rows needed based on the number of numeric columns and the number of columns in the grid
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Ceiling division to ensure all columns fit

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()  # Flatten in case there is only one row

    # Loop through each numeric column and plot the histogram
    for i, col in enumerate(numeric_columns):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram for {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()