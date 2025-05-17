# scripts/generate_embeddings.py
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import uuid
# import random # Not strictly needed here

# Ensure the src directory is accessible for imports
import sys
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
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
        get_text_to_embed as generic_get_text_to_embed, # Rename to avoid conflict if we define a local one
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_CHROMA_PATH,
        DEFAULT_COLLECTION_NAME
    )
    from evaluation.metrics import is_refusal
except ImportError as e:
    print(f"Error importing modules. Make sure src is in Python path: {e}")
    print(f"PROJECT_ROOT={PROJECT_ROOT}, SRC_PATH={SRC_PATH}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64 

def generate_record_id(record: Dict[str, Any], dataset_name: str, index_in_dataset: int) -> str:
    id_keys = ['Index', 'id', 'ID', 'record_id'] 
    found_key = None
    original_id = None
    for key in id_keys:
        if key in record and record[key] is not None:
            found_key = key
            original_id = str(record[key]) 
            break
    if found_key:
        base_id_part = f"{dataset_name}_{found_key}_{original_id}".replace(" ", "_").replace("/", "_") # Sanitize further
        # For JBB, both structures use 'Index' which might collide if files are merged or IDs repeat across files.
        # For AdvBench, if 'Index' is used and files might be merged, also add UUID.
        if found_key == 'Index' and (dataset_name == 'jbb' or dataset_name == 'advbench'):
             unique_suffix = uuid.uuid4().hex[:8] 
             return f"{base_id_part}_{unique_suffix}"
        return base_id_part
    else:
        unique_suffix = uuid.uuid4().hex[:8]
        return f"{dataset_name}_idx_{index_in_dataset}_{unique_suffix}"

def sanitize_metadata(record: Dict[str, Any], include_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    sanitized = {}
    # Keys that are typically the main content for embedding or large text not suitable for metadata
    default_excluded_keys_from_meta = {'transcript', 'prompt', 'Goal', 'goal', 'target', 'Target', 'text', 'page_content', 'target_response'}

    keys_to_iterate = include_keys if include_keys is not None else record.keys()

    for key in keys_to_iterate:
        if key not in record: continue # If include_keys lists a key not in this record
        value = record[key]

        # If include_keys is provided, we only process those keys.
        # If include_keys is NOT provided, we skip the default_excluded_keys_from_meta.
        if include_keys is None and key in default_excluded_keys_from_meta:
            continue
            
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            continue 
        else: # Attempt to convert other types (like lists of tags) to string
            try:
                sanitized[key] = str(value)
            except Exception:
                logger.debug(f"Could not serialize metadata key '{key}' with value type {type(value)}. Skipping.")
    return sanitized

def get_human_turns_from_transcript(transcript_text: str) -> List[str]:
    human_turns = []
    if transcript_text:
        parts = transcript_text.split("\n\nHuman:")
        for part in parts[1:]: 
            human_content = part.split("\n\nAssistant:")[0].strip()
            if human_content:
                human_turns.append(human_content)
    return human_turns

def main(args):
    start_time = time.time()
    logger.info(f"--- Starting Embedding Generation --- Args: {args}")

    datasets_to_process = args.datasets
    if not args.datasets:
        try:
            datasets_to_process = sorted([d.name for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()])
            if not datasets_to_process:
                 logger.error(f"No dataset subdirectories found in {PROCESSED_DATA_DIR}. Exiting."); sys.exit(1)
            logger.info(f"Processing all found datasets: {', '.join(datasets_to_process)}")
        except FileNotFoundError:
             logger.error(f"Processed data directory not found: {PROCESSED_DATA_DIR}"); sys.exit(1)
    
    try:
        embedding_model_instance = EmbeddingModel(model_name=args.embedding_model)
        vector_store = VectorStore(collection_name=args.collection_name, persist_directory=args.db_path)
        if args.force_reload:
            logger.warning(f"Force reloading collection: {args.collection_name}")
            try:
                vector_store.client.delete_collection(name=args.collection_name)
                vector_store = VectorStore(collection_name=args.collection_name, persist_directory=args.db_path) # Re-init
                logger.info(f"Collection deleted and re-initialized.")
            except Exception as e: logger.warning(f"Could not delete collection (may not exist): {e}")
    except Exception as e: logger.error(f"Initialization failed: {e}", exc_info=True); sys.exit(1)

    total_processed, total_added, total_skipped = 0, 0, 0
    batch_texts, batch_metadatas, batch_ids = [], [], []

    for dataset_name in datasets_to_process:
        logger.info(f"--- Processing dataset: {dataset_name} ---")
        try:
            data = get_data(dataset_name, base_dir=PROCESSED_DATA_DIR, lazy=False) # Not filtering at load time
            logger.info(f"Loaded {len(data)} records from {dataset_name}.")
        except Exception as e: logger.error(f"Load failed for '{dataset_name}': {e}. Skipping.", exc_info=True); continue
        if not data: logger.warning(f"No data for '{dataset_name}'. Skipping."); continue

        for i, record in enumerate(data):
            total_processed += 1
            text_to_embed = None
            metadata_fields_to_include = None 
            original_harmful_intent_for_meta = None
            outcome_for_meta = "UNKNOWN_OUTCOME"

            if dataset_name == 'jbb':
                # Structure 1: "judge_comparison_style"
                if 'prompt' in record and 'goal' in record and ('gpt4_cf' in record or 'human_majority' in record):
                    text_to_embed = record.get('prompt')
                    original_harmful_intent_for_meta = record.get('goal')
                    if str(record.get('gpt4_cf', '0')) == '1': outcome_for_meta = "SUCCESSFUL_JAILBREAK"
                    elif str(record.get('human_majority', '0')) == '1': outcome_for_meta = "SUCCESSFUL_JAILBREAK"
                    else: outcome_for_meta = "REFUSED_OR_FAILED"
                    metadata_fields_to_include = ['Index', 'goal', 'prompt', 'target_response', 'human1', 'human2', 'human3', 
                                                'human_majority', 'harmbench_cf', 'gpt4_cf', 'llamaguard2_cf', 'llama3_cf']
                # Structure 2: "harmful_behaviors" (AdvBench-like) style
                elif 'Goal' in record and 'Target' in record: # Simpler check for this structure
                    text_to_embed = record.get('Goal')
                    original_harmful_intent_for_meta = record.get('Goal')
                    target_resp = record.get('Target')
                    if target_resp and not is_refusal(target_resp): outcome_for_meta = "SUCCESSFUL_JAILBREAK"
                    else: outcome_for_meta = "REFUSED_OR_FAILED"
                    metadata_fields_to_include = ['Index', 'Goal', 'Target', 'Behavior', 'Category', 'Source']
                else: # Fallback for JBB if unrecognized
                    text_to_embed = generic_get_text_to_embed(record)
                    original_harmful_intent_for_meta = record.get('goal') or record.get('Goal') or record.get('prompt')

            elif dataset_name == 'advbench': # Handles: {"prompt": ..., "target": ...}
                text_to_embed = record.get('prompt')
                original_harmful_intent_for_meta = record.get('prompt')
                target_resp = record.get('target')
                if target_resp and not is_refusal(target_resp): outcome_for_meta = "SUCCESSFUL_JAILBREAK"
                else: outcome_for_meta = "REFUSED_OR_FAILED"
                # For AdvBench, if it only has 'prompt' and 'target', metadata_fields_to_include can be None.
                # Any other existing fields like 'Index' will be automatically picked up by sanitize_metadata if not in default_excluded_keys_from_meta.
                metadata_fields_to_include = None 

            elif dataset_name == 'anthropic':
                original_harmful_intent_for_meta = record.get('task_description')
                transcript_text = record.get('transcript', '')
                human_turns = get_human_turns_from_transcript(transcript_text)
                text_to_embed = human_turns[-1] if human_turns else original_harmful_intent_for_meta

                rating_str = str(record.get('rating', '0.0'))
                try: rating = float(rating_str)
                except ValueError: rating = 0.0
                ANTHROPIC_SUCCESS_THRESHOLD = 2.5 
                if rating >= ANTHROPIC_SUCCESS_THRESHOLD: outcome_for_meta = "SUCCESSFUL_JAILBREAK"
                else: outcome_for_meta = "REFUSED_OR_FAILED"
                metadata_fields_to_include = ['min_harmlessness_score_transcript', 'num_params', 'model_type', 
                                            'rating', 'task_description', 'task_descripton_harmlessness_score', 
                                            'red_team_member_id', 'is_upworker', 'tags']
            else: 
                text_to_embed = generic_get_text_to_embed(record)
                original_harmful_intent_for_meta = record.get('goal') or record.get('Goal') or record.get('prompt') or text_to_embed
                metadata_fields_to_include = None

            if not text_to_embed:
                logger.warning(f"Skipping record {i} from {dataset_name} (Index: {record.get('Index', 'N/A')}): No text to embed. Keys: {list(record.keys())}")
                total_skipped += 1; continue

            record_id = generate_record_id(record, dataset_name, i)
            current_metadata = sanitize_metadata(record, include_keys=metadata_fields_to_include)
            current_metadata['source_dataset'] = dataset_name
            if original_harmful_intent_for_meta:
                 current_metadata['original_harmful_intent'] = str(original_harmful_intent_for_meta)
            current_metadata['outcome_classification'] = outcome_for_meta
            if original_harmful_intent_for_meta and 'goal' not in current_metadata and (isinstance(original_harmful_intent_for_meta, str) and original_harmful_intent_for_meta.lower() != 'goal'): # Avoid adding 'goal':'Goal'
                if not ('Goal' in current_metadata and current_metadata.get('Goal') == original_harmful_intent_for_meta): # check if original key was 'Goal'
                    current_metadata['goal'] = str(original_harmful_intent_for_meta)


            batch_texts.append(text_to_embed)
            batch_metadatas.append(current_metadata)
            batch_ids.append(record_id)

            if len(batch_texts) >= args.batch_size:
                logger.info(f"Processing batch of {len(batch_texts)} (Total: {total_processed})")
                try:
                    embeddings = embedding_model_instance.encode(batch_texts, show_progress_bar=False)
                    vector_store.add_embeddings(embeddings, batch_texts, batch_metadatas, batch_ids)
                    total_added += len(batch_texts)
                except Exception as e:
                    logger.error(f"Batch add failed: {e}", exc_info=True); total_skipped += len(batch_texts)
                finally: batch_texts, batch_metadatas, batch_ids = [], [], []
        
        logger.info(f"Finished dataset: {dataset_name}")

    if batch_texts:
        logger.info(f"Processing final batch of {len(batch_texts)}")
        try:
            embeddings = embedding_model_instance.encode(batch_texts, show_progress_bar=False)
            vector_store.add_embeddings(embeddings, batch_texts, batch_metadatas, batch_ids)
            total_added += len(batch_texts)
        except Exception as e: 
            logger.error(f"Final batch add failed: {e}", exc_info=True); total_skipped += len(batch_texts)

    logger.info(f"--- Embedding Generation Complete ---")
    logger.info(f"Processed: {total_processed}, Added: {total_added}, Skipped: {total_skipped}")
    if vector_store and vector_store.collection:
        logger.info(f"Collection size: {vector_store.collection.count()}")
    logger.info(f"Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings.")
    parser.add_argument("--datasets", nargs='+', help="Datasets to process (subfolder names). All if omitted.")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_CHROMA_PATH))
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force-reload", action='store_true', help="Delete existing collection first.")
    args = parser.parse_args()
    main(args)