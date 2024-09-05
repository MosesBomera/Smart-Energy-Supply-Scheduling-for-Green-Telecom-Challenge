import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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