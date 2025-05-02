import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

# Ensure the src directory is accessible for imports
import sys
try:
    # Assumes the script is run from the project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
     # Fallback if __file__ is not defined (e.g., interactive environments)
    PROJECT_ROOT = Path.cwd()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Now import from src
try:
    from data.load import get_data, PROCESSED_DATA_DIR
    from rag.embedding import (
        EmbeddingModel,
        VectorStore,
        get_text_to_embed,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_CHROMA_PATH,
        DEFAULT_COLLECTION_NAME
    )
except ImportError as e:
    print(f"Error importing modules. Make sure src is in Python path: {e}")
    print(f"PROJECT_ROOT={PROJECT_ROOT}, SRC_PATH={SRC_PATH}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_BATCH_SIZE = 64 # Number of records to embed and add at once

# --- Helper Functions ---

def generate_record_id(record: Dict[str, Any], dataset_name: str, index_in_dataset: int) -> str:
    """
    Generates a unique and informative ID for a record.
    Tries common ID/Index keys first. If 'Index' is used for jbb, adds a UUID suffix.
    Falls back to using dataset index.
    """
    id_keys = ['Index', 'id', 'ID', 'record_id'] # Check 'Index' first for jbb
    found_key = None
    original_id = None

    for key in id_keys:
        if key in record and record[key] is not None:
            found_key = key
            original_id = str(record[key]) # Convert to string
            break

    if found_key:
        base_id_part = f"{dataset_name}_{found_key}_{original_id}".replace(" ", "_")
        # *** ADD UUID FOR 'Index' key specifically, or any key known to collide ***
        # Adjust the condition if other keys collide
        if found_key == 'Index' and dataset_name == 'jbb':
             unique_suffix = uuid.uuid4().hex[:8] # Add 8 hex chars from UUID
             logger.debug(f"Adding UUID suffix for potentially duplicate key '{found_key}' in dataset '{dataset_name}'. Base: {base_id_part}")
             return f"{base_id_part}_{unique_suffix}"
        else:
             # For other keys assumed unique within their dataset, no suffix needed (or add one if unsure)
             return base_id_part
    else:
        # Fallback to index within the loaded dataset if no explicit ID found
        logger.debug(f"No standard ID key found in record from {dataset_name}. Using index: {index_in_dataset}")
        # Fallback IDs are likely unique per run, but add UUID for safety if mixing runs later? Safer to add.
        unique_suffix = uuid.uuid4().hex[:8]
        return f"{dataset_name}_idx_{index_in_dataset}_{unique_suffix}"

def sanitize_metadata(record: Dict[str, Any], include_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Cleans metadata, ensuring values are compatible with ChromaDB (str, int, float, bool).
    Optionally selects only specific keys.
    Excludes large text fields that are likely embedded separately.
    """
    sanitized = {}
    excluded_keys = {'transcript', 'prompt', 'Goal', 'goal', 'target', 'Target', 'text', 'page_content'} # Fields likely used for embedding content

    keys_to_process = include_keys if include_keys else record.keys()

    for key, value in record.items():
        if key in keys_to_process and key not in excluded_keys:
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                continue # Skip None values
            else:
                # Attempt to convert other types (e.g., list of tags) to string
                try:
                    sanitized[key] = str(value)
                    logger.debug(f"Converted metadata key '{key}' to string: '{str(value)[:50]}...'")
                except Exception:
                    logger.warning(f"Could not serialize metadata key '{key}' with value type {type(value)}. Skipping.")
    return sanitized

# --- Main Script Logic ---

def main(args):
    """Main function to generate embeddings."""
    start_time = time.time()
    logger.info("--- Starting Embedding Generation ---")
    logger.info(f"Arguments: {args}")

    # --- Determine Datasets to Process ---
    if not args.datasets:
        # If no datasets specified, find all subdirectories in processed data dir
        try:
            datasets_to_process = sorted([d.name for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()])
            if not datasets_to_process:
                 logger.error(f"No dataset subdirectories found in {PROCESSED_DATA_DIR}. Exiting.")
                 sys.exit(1)
            logger.info(f"No specific datasets provided, found: {', '.join(datasets_to_process)}")
        except FileNotFoundError:
             logger.error(f"Processed data directory not found: {PROCESSED_DATA_DIR}")
             sys.exit(1)
    else:
        datasets_to_process = args.datasets
        logger.info(f"Processing specified datasets: {', '.join(datasets_to_process)}")


    # --- Initialize Components ---
    try:
        logger.info(f"Initializing embedding model: {args.embedding_model}")
        embedding_model = EmbeddingModel(model_name=args.embedding_model)

        logger.info(f"Initializing vector store: path={args.db_path}, collection={args.collection_name}")
        vector_store = VectorStore(
            collection_name=args.collection_name,
            persist_directory=args.db_path,
            embedding_model=None, # We provide embeddings, not needed here unless for search testing
            embedding_function=None # We provide embeddings
        )

        # Handle force reloading (deleting existing collection)
        if args.force_reload:
            logger.warning(f"Force reload requested. Attempting to delete collection: {args.collection_name}")
            try:
                vector_store.client.delete_collection(name=args.collection_name)
                logger.info(f"Successfully deleted collection: {args.collection_name}")
                # Re-initialize VectorStore instance after deletion
                vector_store = VectorStore(
                    collection_name=args.collection_name,
                    persist_directory=args.db_path,
                    embedding_model=None,
                    embedding_function=None
                )
                logger.info(f"Re-initialized vector store after deletion.")
            except Exception as e:
                # Catch potential errors if collection didn't exist, log and continue
                logger.warning(f"Could not delete collection (may not exist): {e}")

    except Exception as e:
        logger.error(f"Failed to initialize embedding model or vector store: {e}", exc_info=True)
        sys.exit(1)

    # --- Process Data ---
    total_processed = 0
    total_added = 0
    total_skipped = 0

    batch_texts: List[str] = []
    batch_metadatas: List[Dict[str, Any]] = []
    batch_ids: List[str] = []

    for dataset_name in datasets_to_process:
        logger.info(f"--- Processing dataset: {dataset_name} ---")
        try:
            # Load entire dataset into memory (necessary for batching)
            # If datasets are HUGE, need a different strategy (lazy loading + smaller batches)
            data = get_data(dataset_name, base_dir=PROCESSED_DATA_DIR, lazy=False)
            logger.info(f"Loaded {len(data)} records from {dataset_name}.")
        except FileNotFoundError:
            logger.error(f"Dataset directory or file not found for '{dataset_name}'. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Failed to load data for '{dataset_name}': {e}. Skipping.", exc_info=True)
            continue

        if not data:
            logger.warning(f"No data loaded for dataset '{dataset_name}'. Skipping.")
            continue

        for i, record in enumerate(data):
            total_processed += 1

            # 1. Get text to embed
            text_to_embed = get_text_to_embed(record)

            if not text_to_embed:
                logger.debug(f"Skipping record {i} from {dataset_name}: No suitable text found.")
                total_skipped += 1
                continue

            # 2. Generate ID
            record_id = generate_record_id(record, dataset_name, i)

            # 3. Prepare Metadata
            metadata = sanitize_metadata(record)
            metadata['source_dataset'] = dataset_name # Ensure source dataset is tracked

            # 4. Add to batch
            batch_texts.append(text_to_embed)
            batch_metadatas.append(metadata)
            batch_ids.append(record_id)

            # 5. Process batch if full
            if len(batch_texts) >= args.batch_size:
                logger.info(f"Processing batch of {len(batch_texts)} records (Total processed: {total_processed})...")
                try:
                    embeddings = embedding_model.encode(batch_texts, show_progress_bar=False) # Progress bar can be noisy
                    vector_store.add_embeddings(embeddings, batch_texts, batch_metadatas, batch_ids)
                    total_added += len(batch_texts)
                    logger.debug(f"Added batch. Total added so far: {total_added}")
                except Exception as e:
                    logger.error(f"Failed to process or add batch: {e}. Skipping this batch.", exc_info=True)
                    total_skipped += len(batch_texts) # Count skipped batch items
                finally:
                    # Clear batch regardless of success/failure
                    batch_texts, batch_metadatas, batch_ids = [], [], []

        logger.info(f"Finished processing records for dataset: {dataset_name}")

    # --- Process final leftover batch ---
    if batch_texts:
        logger.info(f"Processing final batch of {len(batch_texts)} records...")
        try:
            embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
            vector_store.add_embeddings(embeddings, batch_texts, batch_metadatas, batch_ids)
            total_added += len(batch_texts)
            logger.info(f"Added final batch. Total added: {total_added}")
        except Exception as e:
            logger.error(f"Failed to process or add final batch: {e}", exc_info=True)
            total_skipped += len(batch_texts)

    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info("--- Embedding Generation Complete ---")
    logger.info(f"Total records processed: {total_processed}")
    logger.info(f"Total records added to vector store: {total_added}")
    logger.info(f"Total records skipped: {total_skipped}")
    logger.info(f"Final collection size: {vector_store.collection.count()}")
    logger.info(f"Total time taken: {duration:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for datasets and store them in ChromaDB.")

    parser.add_argument(
        "--datasets",
        nargs='+',
        help="List of dataset names (subfolder names in data/processed) to process. If omitted, processes all found datasets."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Name of the sentence-transformer model to use (default: {DEFAULT_EMBEDDING_MODEL})."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_CHROMA_PATH),
        help=f"Path to the ChromaDB persistence directory (default: {DEFAULT_CHROMA_PATH})."
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Name of the ChromaDB collection (default: {DEFAULT_COLLECTION_NAME})."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of records to embed and add in each batch (default: {DEFAULT_BATCH_SIZE})."
    )
    parser.add_argument(
        "--force-reload",
        action='store_true',
        help="If set, delete the existing collection before starting embedding generation."
    )

    args = parser.parse_args()
    main(args)