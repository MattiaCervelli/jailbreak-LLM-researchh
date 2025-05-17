# src/data/load.py

import json
import os
import random
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Union, Any, Generator, Iterable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory for processed data relative to this file's location
# Assumes this script is in /src/data/ and processed data is in /data/processed/
# Adjust the number of .parent calls if the script location changes
try:
    # This assumes the script is run from the project root or within src/
    # If running src/data/load.py directly, Path.cwd() might be different
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Handle cases where __file__ is not defined (e.g., interactive environments)
    # Fallback to current working directory, assuming it's the project root
    PROJECT_ROOT = Path.cwd()
    logging.warning(f"__file__ not defined. Assuming project root is current working directory: {PROJECT_ROOT}")


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
if not PROCESSED_DATA_DIR.is_dir():
     # Fallback if the standard structure isn't found relative to script
    alt_path = Path("data") / "processed"
    if alt_path.is_dir():
        PROCESSED_DATA_DIR = alt_path
    else:
        # Raise an error if data dir cannot be found, preventing cryptic downstream errors
        raise FileNotFoundError(
            f"Processed data directory not found at {PROCESSED_DATA_DIR} or {alt_path}. "
            "Please ensure the directory exists and contains processed JSONL datasets, "
            "or adjust PROCESSED_DATA_DIR path in src/data/load.py"
        )
logging.info(f"Using processed data directory: {PROCESSED_DATA_DIR}")

# --- Core Loading Functions ---

def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Loads a JSONL file into a list of dictionaries.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A list where each element is a dictionary parsed from a line in the JSONL file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Skip empty lines
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON on line {i+1} in file {file_path}: {e}")
                    logging.error(f"Problematic line: {line.strip()}")
                    # Option 1: Skip corrupted line (more robust)
                    continue
                    # Option 2: Raise the error (stops execution)
                    # raise json.JSONDecodeError(f"Error on line {i+1} in {file_path}: {e.msg}", e.doc, e.pos)
        return data
    except Exception as e:
        logging.error(f"Failed to read or parse file {file_path}: {e}")
        raise # Re-raise the exception after logging

def load_jsonl_lazy(file_path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    """
    Lazily loads a JSONL file, yielding one dictionary at a time.
    Useful for very large files that may not fit into memory.

    Args:
        file_path: Path to the JSONL file.

    Yields:
        A dictionary parsed from a line in the JSONL file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON (and not skipped).
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                 # Skip empty lines
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON on line {i+1} in file {file_path}: {e}")
                    logging.error(f"Problematic line: {line.strip()}")
                    # Option 1: Skip corrupted line
                    continue
                    # Option 2: Raise the error
                    # raise json.JSONDecodeError(f"Error on line {i+1} in {file_path}: {e.msg}", e.doc, e.pos)
    except Exception as e:
        logging.error(f"Failed to read or parse file {file_path}: {e}")
        raise

def load_datasets(
    dataset_names: Union[str, List[str]],
    base_dir: Path = PROCESSED_DATA_DIR,
    lazy: bool = False
) -> Union[List[Dict[str, Any]], Generator[Dict[str, Any], None, None]]:
    """
    Loads one or more datasets from their respective subdirectories under base_dir.
    Looks for all .jsonl files within each specified dataset directory.

    Args:
        dataset_names: A single dataset name (e.g., 'anthropic') or a list of names.
                       These should correspond to subdirectory names in base_dir.
        base_dir: The root directory containing the dataset subdirectories.
                  Defaults to PROCESSED_DATA_DIR.
        lazy: If True, returns a generator yielding records one by one.
              If False (default), loads all data into a list in memory.

    Returns:
        If lazy=False: A list containing all records from the specified datasets.
        If lazy=True: A generator yielding all records from the specified datasets.

    Raises:
        FileNotFoundError: If a specified dataset directory does not exist.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    if lazy:
        return _load_datasets_lazy(dataset_names, base_dir)
    else:
        all_data = []
        for name in dataset_names:
            dataset_dir = Path(base_dir) / name
            if not dataset_dir.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

            logging.info(f"Loading data from: {dataset_dir}")
            jsonl_files = list(dataset_dir.glob('*.jsonl'))
            if not jsonl_files:
                 logging.warning(f"No .jsonl files found in {dataset_dir}")
                 continue

            for file_path in jsonl_files:
                logging.debug(f"Loading file: {file_path}")
                all_data.extend(load_jsonl(file_path))

        logging.info(f"Loaded total {len(all_data)} records from datasets: {', '.join(dataset_names)}")
        return all_data

def _load_datasets_lazy(
    dataset_names: List[str],
    base_dir: Path
) -> Generator[Dict[str, Any], None, None]:
    """Helper generator function for lazy loading."""
    total_yielded = 0
    for name in dataset_names:
        dataset_dir = Path(base_dir) / name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        logging.info(f"Streaming data from: {dataset_dir}")
        jsonl_files = list(dataset_dir.glob('*.jsonl'))
        if not jsonl_files:
            logging.warning(f"No .jsonl files found in {dataset_dir}")
            continue

        for file_path in jsonl_files:
            logging.debug(f"Streaming file: {file_path}")
            try:
                for record in load_jsonl_lazy(file_path):
                    yield record
                    total_yielded += 1
            except Exception as e:
                 logging.error(f"Error streaming from {file_path}: {e}. Skipping rest of file.")
                 continue # Skip to next file if one is corrupted

    logging.info(f"Finished streaming data. Total records yielded (approx): {total_yielded} from datasets: {', '.join(dataset_names)}")


# --- Filtering Functions ---

def filter_data(
    data: Iterable[Dict[str, Any]],
    criteria: Optional[Dict[str, Any]] = None
) -> Union[List[Dict[str, Any]], Generator[Dict[str, Any], None, None]]:
    """
    Filters a list or generator of dictionaries based on specified criteria.

    Args:
        data: An iterable (list or generator) of dictionaries (records).
        criteria: A dictionary where keys are field names and values are the
                  desired values to filter by. A record must match ALL criteria.
                  If None or empty, no filtering is applied.

    Returns:
        If input is a list: A new list containing only the matching records.
        If input is a generator: A new generator yielding only matching records.
    """
    if not criteria:
        return data # Return original if no criteria

    def _matches(record: Dict[str, Any]) -> bool:
        """Checks if a single record matches all criteria."""
        for key, value in criteria.items():
            # Use .get() for safety in case a key is missing in some records
            if record.get(key) != value:
                return False
        return True

    if isinstance(data, Generator):
        # Return a generator if the input was a generator
        return (record for record in data if _matches(record))
    else:
        # Assume input is a list or other concrete iterable
        return [record for record in data if _matches(record)]


# --- Sampling Functions ---

def sample_random(
    data: List[Dict[str, Any]],
    n: int,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Selects a random sample of n items from the data.

    Args:
        data: A list of dictionaries (records).
        n: The number of samples to select.
        random_seed: Optional seed for the random number generator for reproducibility.

    Returns:
        A new list containing n randomly selected records. Returns the original list
        shuffled if n >= len(data).
    """
    if random_seed is not None:
        random.seed(random_seed)

    if n >= len(data):
        logging.warning(f"Requested sample size {n} >= data size {len(data)}. Returning shuffled data.")
        shuffled_data = data[:] # Create a copy
        random.shuffle(shuffled_data)
        return shuffled_data

    return random.sample(data, n)

def sample_balanced(
    data: List[Dict[str, Any]],
    n: int,
    group_by_key: str,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Selects a sample of n items, attempting to balance across categories
    defined by the `group_by_key`.

    Handles groups with fewer items than the target per group. Prioritizes
    getting samples from all groups before taking extras from larger groups.

    Args:
        data: A list of dictionaries (records).
        n: The total number of samples desired.
        group_by_key: The dictionary key whose values define the groups for balancing.
        random_seed: Optional seed for the random number generator for reproducibility.

    Returns:
        A new list containing n records, sampled with balance across groups.
        Returns fewer than n items if the total data is less than n.
    """
    if n <= 0:
        return []
    if not data:
        return []
    if n >= len(data):
        logging.warning(f"Requested sample size {n} >= data size {len(data)}. Returning shuffled data.")
        if random_seed is not None:
            random.seed(random_seed)
        shuffled_data = data[:] # Create a copy
        random.shuffle(shuffled_data)
        return shuffled_data

    if random_seed is not None:
        random.seed(random_seed)

    # Group data by the specified key
    groups = defaultdict(list)
    for record in data:
        key_value = record.get(group_by_key)
        # Handle potential missing keys if needed, e.g., group under None
        # if key_value is None: key_value = "None" # Or skip? Depends on desired behavior
        groups[key_value].append(record)

    num_groups = len(groups)
    if num_groups == 0:
        logging.warning(f"No groups found for key '{group_by_key}'. Performing random sampling instead.")
        return sample_random(data, n, random_seed) # Fallback to random

    # Shuffle items within each group for random selection within group
    for key in groups:
        random.shuffle(groups[key])

    # Calculate base samples per group and remainder
    base_samples_per_group = n // num_groups
    remainder = n % num_groups

    sampled_data = []
    group_keys = list(groups.keys())
    random.shuffle(group_keys) # Shuffle group order to randomly distribute remainder

    # First pass: Take up to base_samples_per_group from each group
    remaining_items = {} # Store items not yet taken for the remainder pass
    for key in group_keys:
        group_items = groups[key]
        count = min(len(group_items), base_samples_per_group)
        sampled_data.extend(group_items[:count])
        remaining_items[key] = group_items[count:] # Store the rest

    # Second pass: Distribute the remainder among groups that still have items
    groups_with_remaining = [k for k in group_keys if remaining_items[k]]
    random.shuffle(groups_with_remaining) # Shuffle again for remainder distribution

    items_needed = n - len(sampled_data) # Should equal remainder initially
    items_added_remainder = 0

    # Distribute remainder round-robin style
    group_idx = 0
    while items_added_remainder < items_needed and groups_with_remaining:
        current_group_key = groups_with_remaining[group_idx % len(groups_with_remaining)]
        if remaining_items[current_group_key]:
            sampled_data.append(remaining_items[current_group_key].pop(0)) # Take one item
            items_added_remainder += 1
            # If group is now empty for remainder, remove it from consideration (optional optimization)
            if not remaining_items[current_group_key]:
                 # Find the key and remove it efficiently if list is potentially large
                 # For simplicity here, we just let the modulo wrap around;
                 # the check `if remaining_items[current_group_key]:` handles exhaustion.
                 pass # Or remove from groups_with_remaining list if efficiency needed
        group_idx += 1
        # Safety break to prevent infinite loop if logic is flawed or items run out unexpectedly
        if group_idx > num_groups * 2 and items_added_remainder < items_needed :
             logging.warning("Balanced sampling remainder distribution stopped potentially early.")
             break


    # Final shuffle of the entire sampled list
    random.shuffle(sampled_data)

    if len(sampled_data) != n:
         logging.warning(f"Could only sample {len(sampled_data)} items due to data/group constraints, requested {n}.")

    return sampled_data


# --- Main Interface Function ---

def get_data(
    dataset_names: Union[str, List[str]],
    n_samples: Optional[int] = None,
    filter_criteria: Optional[Dict[str, Any]] = None,
    balance_key: Optional[str] = None,
    random_seed: Optional[int] = None,
    base_dir: Path = PROCESSED_DATA_DIR,
    lazy: bool = False
) -> Union[List[Dict[str, Any]], Generator[Dict[str, Any], None, None]]:
    """
    High-level function to load, filter, and sample data from specified datasets.

    Args:
        dataset_names: Name or list of names of datasets to load (e.g., 'anthropic', 'advbench').
        n_samples: Total number of samples desired. If None, returns all filtered data.
        filter_criteria: Dictionary for filtering records (e.g., {'technique': 'X'}).
        balance_key: Key to use for balanced sampling (e.g., 'technique', 'category').
                     If None, random sampling is used (if n_samples is set).
        random_seed: Seed for random operations (shuffling, sampling) for reproducibility.
        base_dir: Root directory containing dataset subdirectories.
        lazy: If True, filtering is applied lazily, and sampling is NOT possible.
              Returns a generator. n_samples and balance_key are ignored if lazy=True.

    Returns:
        If lazy=True: A generator yielding filtered records.
        If lazy=False: A list of dictionaries representing the final dataset
                       (loaded, filtered, and potentially sampled/balanced).

    Raises:
        ValueError: If lazy=True is combined with sampling options (n_samples or balance_key).
        FileNotFoundError: If specified dataset directories or files are not found.
    """
    if lazy and (n_samples is not None or balance_key is not None):
        raise ValueError("Lazy loading (lazy=True) cannot be combined with sampling (n_samples) or balancing (balance_key). "
                         "Load data into memory first (lazy=False) for these operations.")

    if random_seed is not None:
        random.seed(random_seed)
        logging.info(f"Using random seed: {random_seed}")

    # Step 1: Load data (potentially lazy)
    data = load_datasets(dataset_names, base_dir=base_dir, lazy=lazy)
    logging.info(f"Initial loading step complete for datasets: {dataset_names}. Lazy={lazy}.")

    # Step 2: Filter data (works on both list and generator)
    if filter_criteria:
        logging.info(f"Applying filter criteria: {filter_criteria}")
        data = filter_data(data, criteria=filter_criteria)
        # Note: If lazy, data is now a filter generator. If not lazy, it's a filtered list.
    else:
         logging.info("No filter criteria applied.")


    # --- Steps below require data in memory (lazy=False) ---
    if lazy:
        # If lazy loading was requested, return the (potentially filtered) generator
        return data # Type: Generator[Dict[str, Any], None, None]

    # If not lazy, data is now a list (or was loaded as one)
    # Ensure data is a list before sampling
    if isinstance(data, Generator): # Should not happen if lazy=False, but as safety check
         logging.warning("Data was unexpectedly a generator after non-lazy load/filter. Converting to list.")
         data = list(data)

    # Log size after filtering, before sampling
    logging.info(f"Data size after filtering (before sampling): {len(data)}")


    # Step 3: Sample data (if requested)
    if n_samples is not None:
        if n_samples <= 0 :
             logging.warning(f"Requested sample size {n_samples} <= 0. Returning empty list.")
             return []

        if balance_key:
            logging.info(f"Performing balanced sampling: {n_samples} samples, balancing by '{balance_key}'")
            final_data = sample_balanced(data, n_samples, balance_key, random_seed)
        else:
            logging.info(f"Performing random sampling: {n_samples} samples")
            final_data = sample_random(data, n_samples, random_seed)
        logging.info(f"Final data size after sampling: {len(final_data)}")
    else:
        # No sampling requested, return all filtered data
        final_data = data
        logging.info(f"No sampling requested. Returning all {len(final_data)} filtered records.")


    return final_data