import logging
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import shutil

def download_file(directory: Union[str, Path], source_path: Union[str, Path], decompress: bool = False) -> None:
    """Copy file from source_path to directory.

    Parameters
    ----------
    directory: str, Path
        Custom directory where data will be copied to.
    source_path: str, Path
        Path where the file is located.
    decompress: bool
        Whether to decompress the copied file. Default False.
    """
    # Convert directory and source_path to Path objects
    directory = Path(directory)
    source_path = Path(source_path)
    
    # Ensure the destination directory exists
    directory.mkdir(parents=True, exist_ok=True)
    
    # Calculate destination file path
    filename = source_path.name
    destination_path = directory / filename
    
    # Copy the file from source to destination
    shutil.copy2(source_path, destination_path)
    
    # Log success message
    size = destination_path.stat().st_size
    logger.info(f"Successfully copied {filename}, {size} bytes.")
    
    # Optionally decompress the file (not needed in most cases for CSV files)
    if decompress:
        if destination_path.suffix == '.zip':
            import zipfile
            with zipfile.ZipFile(destination_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            logger.info(f"Successfully decompressed {destination_path}")
# Cell
@dataclass
class Info:
    """
    Info Dataclass of datasets.
    Args:
        groups (Tuple): Tuple of str groups
        class_groups (Tuple): Tuple of dataclasses.
    """
    groups: Tuple[str]
    class_groups: Tuple[dataclass]

    def get_group(self, group: str):
        """Gets dataclass of group."""
        if group not in self.groups:
            raise Exception(f'Unkown group {group}')

        return self.class_groups[self.groups.index(group)]

    def __getitem__(self, group: str):
        """Gets dataclass of group."""
        if group not in self.groups:
            raise Exception(f'Unkown group {group}')

        return self.class_groups[self.groups.index(group)]

    def __iter__(self):
        for group in self.groups:
            yield group, self.get_group(group)


# Cell
@dataclass
class TimeSeriesDataclass:
    """
    Args:
        S (pd.DataFrame): DataFrame of static features of shape
            (n_time_series, n_features).
        X (pd.DataFrame): DataFrame of exogenous variables of shape
            (sum n_periods_i for i=1..n_time_series, n_exogenous).
        Y (pd.DataFrame): DataFrame of target variable of shape
            (sum n_periods_i for i=1..n_time_series, 1).
        idx_categorical_static (list, optional): List of categorical indexes
            of S.
        group (str, optional): Group name if applies.
            Example: 'Yearly'
    """
    S: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    idx_categorical_static: Optional[List] = None
    group: Union[str, List[str]] = None