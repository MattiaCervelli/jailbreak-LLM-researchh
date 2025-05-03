import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import random # Import random for varying seeds

# --- (Keep existing imports and path setup) ---
import sys
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

SRC_PATH = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "configs" # Define path to configs
RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True) # Ensure results directory exists

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from rag.embedding import EmbeddingModel, VectorStore, DEFAULT_EMBEDDING_MODEL, DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME
    from models.local_llm import OllamaLLM, DEFAULT_MODEL as DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL
    from attacks.generator import AttackGenerator, DEFAULT_PROMPT_TEMPLATES_PATH, DEFAULT_RETRIEVAL_K
except ImportError as e:
    print(f"Error importing modules. Make sure src is in Python path: {e}")
    print(f"PROJECT_ROOT={PROJECT_ROOT}, SRC_PATH={SRC_PATH}")
    sys.exit(1)

# --- (Keep existing logging setup) ---
log_file_path = RESULTS_PATH / f"generation_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout) # Also print logs to console
    ]
)
logger = logging.getLogger(__name__)

# --- Default Configuration ---
# No longer use default seeds here, will load from file
DEFAULT_TECHNIQUES = ['generic', 'role_play', 'hypothetical']
DEFAULT_N_CANDIDATES = 1 # Default to generating 1 candidate
DEFAULT_OUTPUT_FILE = RESULTS_PATH / f"generated_attacks_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
DEFAULT_SEED_FILE_PATH = CONFIG_PATH / "seed_queries.txt" # Default path for seed file

# --- Helper Functions ---

def load_seed_queries(file_path: Union[str, Path]) -> List[str]:
    """Loads seed queries from a file (one query per line)."""
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"Seed query file not found: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, filter out empty lines
            seeds = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')] # Ignore empty lines and comments
        logger.info(f"Loaded {len(seeds)} seed queries from {path}.")
        return seeds
    except Exception as e:
        logger.error(f"Error loading seed queries from {path}: {e}")
        return []

# --- (Keep format_result_for_saving function) ---
def format_result_for_saving(
    result: Dict[str, Any],
    run_id: str,
    seed_index: int,
    tech_index: int,
    candidate_index: int # Added candidate index
    ) -> Dict[str, Any]:
    """Selects and formats fields from the generation result for saving."""
    # Include candidate index in the generation_id for uniqueness
    generation_id = f"gen_{seed_index}_{tech_index}_{candidate_index}_{uuid.uuid4().hex[:6]}"

    formatted = {
        "run_id": run_id,
        "generation_id": generation_id,
        "seed_index": seed_index, # Keep track of original seed
        "technique_index": tech_index, # Keep track of technique order
        "candidate_index": candidate_index, # Keep track of which candidate this is (0 to N-1)
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "seed_query": result.get("seed_query"),
        "technique": result.get("technique"),
        "generation_success": result.get("success"),
        "error": result.get("error"),
        "generated_attack": result.get("generated_attack_cleaned"),
        # Optionally include more details below
        "generated_attack_raw": result.get("generated_attack_raw"),
        "topic_hint": result.get("topic_hint"),
    }

    retrieved = result.get("retrieved_examples")
    if retrieved and retrieved.get('ids'):
        examples = []
        # Limit stored examples summary to avoid excessive file size, e.g., top 3
        num_examples_to_log = min(len(retrieved['ids']), 3)
        for i in range(num_examples_to_log):
            example = {
                "id": retrieved['ids'][i],
                "distance": retrieved['distances'][i] if retrieved.get('distances') and i < len(retrieved['distances']) else None,
                "doc_snippet": retrieved['documents'][i][:100] + "..." if retrieved.get('documents') and i < len(retrieved['documents']) else None, # Snippet only
                "metadata": retrieved['metadatas'][i] if retrieved.get('metadatas') and i < len(retrieved['metadatas']) else {}
            }
            examples.append(example)
        formatted["retrieved_examples_summary"] = examples
    else:
        formatted["retrieved_examples_summary"] = None

    return formatted

# --- Main Orchestration Logic ---

def run_generation(args):
    """Runs the attack generation process, supporting N candidates."""
    run_start_time = time.time()
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"--- Starting Attack Generation Run (Best-of-{args.n_candidates}) ---")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Output will be saved to: {args.output_file}")
    logger.info(f"Logs will be saved to: {log_file_path}")

    # --- Load Seeds and Techniques ---
    seed_queries = load_seed_queries(args.seed_file) # Load from specified file
    techniques = args.techniques or DEFAULT_TECHNIQUES
    if not seed_queries:
        logger.error("No seed queries loaded. Exiting.")
        sys.exit(1)
    logger.info(f"Using techniques: {', '.join(techniques)}")
    logger.info(f"Generating {args.n_candidates} candidates per seed/technique.")

    # --- Initialize Components ---
    # ...(Keep component initialization block)...
    logger.info("Initializing components...")
    try:
        query_embedder = EmbeddingModel(model_name=args.embedding_model)
        vector_store = VectorStore(
            collection_name=args.collection_name,
            persist_directory=args.db_path
        )
        if vector_store.collection.count() == 0:
             logger.warning(f"Vector store at {args.db_path} (collection: {args.collection_name}) is empty!")
        generator_llm = OllamaLLM(
            model_name=args.generator_llm,
            ollama_url=args.ollama_url
        )
        attack_generator = AttackGenerator(
            retriever=vector_store,
            query_embedder=query_embedder,
            generator_llm=generator_llm,
            prompt_templates_path=args.templates_file,
            default_k=args.k
        )
        logger.info("Components initialized successfully.")
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Generation Loop ---
    # ...(Keep calculation of total_base_attempts, total_candidates_to_generate, counters)...
    total_base_attempts = len(seed_queries) * len(techniques)
    total_candidates_to_generate = total_base_attempts * args.n_candidates
    generated_candidate_count = 0
    failed_candidate_count = 0
    logger.info(f"Starting generation for {total_base_attempts} base attempts ({total_candidates_to_generate} total candidates)...")


    # ...(Keep the rest of the generation loop, including the inner N-candidate loop and saving logic)...
    try:
        with open(args.output_file, 'a', encoding='utf-8') as f_out:
            for i, seed in enumerate(seed_queries):
                for j, tech in enumerate(techniques):
                    base_attempt_num = i * len(techniques) + j + 1
                    logger.info(f"--- Base Attempt {base_attempt_num}/{total_base_attempts}: Seed='{seed[:50]}...', Technique='{tech}' ---")

                    # Inner loop for N candidates
                    for n in range(args.n_candidates):
                        candidate_num_total = (base_attempt_num - 1) * args.n_candidates + n + 1
                        logger.info(f" Generating Candidate {n+1}/{args.n_candidates} (Total: {candidate_num_total}/{total_candidates_to_generate})")

                        try:
                            # Modify generation options for diversity, primarily using seed
                            current_gen_opts = {'temperature': args.temperature}
                            current_gen_opts['seed'] = random.randint(0, 2**32 - 1) # Generate a random seed
                            logger.debug(f" Using generation options: {current_gen_opts}")

                            result = attack_generator.generate_attack(
                                seed_query=seed,
                                technique=tech,
                                k=args.k,
                                retrieval_filter=None,
                                generation_options=current_gen_opts # Pass options with varying seed
                            )

                            if result["success"]:
                                generated_candidate_count += 1
                            else:
                                failed_candidate_count += 1

                            # Format and save result, including candidate index
                            save_data = format_result_for_saving(result, run_id, i, j, n) # Pass 'n'
                            f_out.write(json.dumps(save_data) + '\n')

                        except Exception as e:
                            failed_candidate_count += 1
                            logger.error(f"Critical error during generation for seed '{seed[:50]}...', technique '{tech}', candidate {n+1}: {e}", exc_info=True)
                            # Save error record
                            error_data = {
                                "run_id": run_id,
                                "generation_id": f"gen_{i}_{j}_{n}_{uuid.uuid4().hex[:6]}", # Include candidate index
                                "seed_index": i,
                                "technique_index": j,
                                "candidate_index": n,
                                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                                "seed_query": seed,
                                "technique": tech,
                                "generation_success": False,
                                "error": f"Script-level exception: {e}",
                                "generated_attack": None,
                            }
                            f_out.write(json.dumps(error_data) + '\n')

                        # Optional delay between individual candidate generations (within a base attempt)
                        # if args.delay > 0 and n < args.n_candidates - 1:
                        #     time.sleep(args.delay / args.n_candidates) # Shorter delay maybe?

                    # Optional delay between base attempts (after generating all N candidates for one seed/technique)
                    if args.delay > 0 and base_attempt_num < total_base_attempts:
                        logger.debug(f"Waiting {args.delay}s before next base attempt...")
                        time.sleep(args.delay)

    except IOError as e:
        logger.error(f"Error writing to output file {args.output_file}: {e}")
    except KeyboardInterrupt:
         logger.warning("Generation interrupted by user.")
    except Exception as e:
         logger.error(f"An unexpected error occurred during the generation loop: {e}", exc_info=True)

    # --- (Keep final summary logging) ---
    run_end_time = time.time()
    duration = run_end_time - run_start_time
    logger.info(f"--- Attack Generation Run Complete (Best-of-{args.n_candidates}) ---")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Total base attempts: {total_base_attempts}")
    logger.info(f"Total candidates generated/attempted: {generated_candidate_count + failed_candidate_count} / {total_candidates_to_generate}")
    logger.info(f"Successful generations (candidates): {generated_candidate_count}")
    logger.info(f"Failed generations (candidates): {failed_candidate_count}")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info(f"Detailed logs saved to: {log_file_path}")
    logger.info(f"Total time taken: {duration:.2f} seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG-based attack generation using different techniques and N candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output Args - Changed seed_file default and made it optional
    parser.add_argument("--seed-file", type=str, default=str(DEFAULT_SEED_FILE_PATH), help="Path to a text file containing seed queries (one per line).")
    parser.add_argument("--output-file", type=str, default=str(DEFAULT_OUTPUT_FILE), help="Path to the output JSONL file to save results.")
    parser.add_argument("--techniques", nargs='+', default=DEFAULT_TECHNIQUES, help="List of techniques (keys from templates file) to use.")

    # Model and Path Args
    # ...(Keep other existing arguments)...
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Sentence Transformer model for query embedding.")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_CHROMA_PATH), help="Path to ChromaDB directory.")
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME, help="ChromaDB collection name.")
    parser.add_argument("--generator-llm", type=str, default=DEFAULT_LLM_MODEL, help="Ollama model name for generation (e.g., 'mistral:latest').")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL, help="URL of the Ollama server.")
    parser.add_argument("--templates-file", type=str, default=str(DEFAULT_PROMPT_TEMPLATES_PATH), help="Path to the prompt templates YAML file.")
    parser.add_argument("-k", type=int, default=DEFAULT_RETRIEVAL_K, help="Number of examples to retrieve for RAG.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the generator LLM.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between *base* generation attempts (after N candidates). Set 0 to disable.")
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES, help="Number of candidate prompts to generate for each seed/technique.")


    args = parser.parse_args()

    if args.n_candidates < 1:
        parser.error("--n-candidates must be at least 1.")

    # Check if the specified seed file exists if provided
    if args.seed_file and not Path(args.seed_file).is_file():
         parser.error(f"Specified seed file not found: {args.seed_file}")

    run_generation(args)