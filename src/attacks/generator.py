import logging
import time
import yaml # Import YAML
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# --- (Keep existing imports and setup) ---
# Ensure the src directory is accessible for imports
import sys
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

SRC_PATH = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "configs" # Define path to configs
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from rag.embedding import EmbeddingModel, VectorStore, DEFAULT_EMBEDDING_MODEL, DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME
    from models.local_llm import OllamaLLM, DEFAULT_MODEL as DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL
except ImportError as e:
    print(f"Error importing modules. Make sure src is in Python path: {e}")
    print(f"PROJECT_ROOT={PROJECT_ROOT}, SRC_PATH={SRC_PATH}")
    sys.exit(1)

# --- (Keep logging setup) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_RETRIEVAL_K = 5
DEFAULT_PROMPT_TEMPLATES_PATH = CONFIG_PATH / "prompt_templates.yaml" # Path to templates

class AttackGenerator:
    """
    Generates jailbreak attack prompts using RAG (Retrieval-Augmented Generation)
    with a local LLM, supporting different techniques via prompt templates.
    """

    def __init__(
        self,
        retriever: VectorStore,
        query_embedder: EmbeddingModel,
        generator_llm: OllamaLLM,
        prompt_templates_path: Union[str, Path] = DEFAULT_PROMPT_TEMPLATES_PATH,
        default_k: int = DEFAULT_RETRIEVAL_K,
    ):
        """
        Initializes the AttackGenerator.

        Args:
            retriever: An initialized VectorStore instance for retrieving examples.
            query_embedder: An initialized EmbeddingModel instance used for query embedding.
            generator_llm: An initialized OllamaLLM instance for generating the attack.
            prompt_templates_path: Path to the YAML file containing prompt templates.
            default_k: Default number of examples to retrieve for RAG.
        """
        # --- (Keep type checks for retriever, query_embedder, generator_llm) ---
        if not isinstance(retriever, VectorStore):
             raise TypeError("Retriever must be an instance of VectorStore")
        if not isinstance(query_embedder, EmbeddingModel):
             raise TypeError("Query embedder must be an instance of EmbeddingModel")
        if not isinstance(generator_llm, OllamaLLM):
            raise TypeError("Generator LLM must be an instance of OllamaLLM")


        self.retriever = retriever
        self.query_embedder = query_embedder
        self.generator_llm = generator_llm
        self.default_k = default_k
        self.prompt_templates = self._load_prompt_templates(prompt_templates_path)

        logger.info("AttackGenerator initialized.")
        # --- (Keep logging other info) ---
        logger.info(f"  Retriever Collection: {self.retriever.collection_name}")
        logger.info(f"  Query Embedder Model: {self.query_embedder.model_name}")
        logger.info(f"  Generator LLM Model: {self.generator_llm.model_name}")
        logger.info(f"  Default K for Retrieval: {self.default_k}")
        logger.info(f"  Loaded prompt templates for techniques: {list(self.prompt_templates.keys())}")


    def _load_prompt_templates(self, path: Union[str, Path]) -> Dict[str, str]:
        """Loads prompt templates from a YAML file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                templates = yaml.safe_load(f)
            if not isinstance(templates, dict):
                raise ValueError("Prompt template file should be a dictionary (YAML mapping).")
            logger.info(f"Successfully loaded prompt templates from: {path}")
            return templates
        except FileNotFoundError:
            logger.error(f"Prompt template file not found at: {path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML template file {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt templates from {path}: {e}")
            raise

    def _format_retrieved_examples(self, search_results: Optional[Dict[str, List[Any]]]) -> str:
        """Formats the retrieved documents for inclusion in the prompt."""
        if not search_results or not search_results.get('documents'):
            return "No relevant examples found in the database."

        formatted = ""
        for i, doc_text in enumerate(search_results['documents']):
            # Add more metadata to the examples if helpful for the generator LLM
            meta = search_results['metadatas'][i] if search_results.get('metadatas') and i < len(search_results['metadatas']) else {}
            source = meta.get('source_dataset', 'unknown')
            # Maybe include technique if available in metadata? technique = meta.get('technique', 'unknown')
            formatted += f"{i+1}. [Source: {source}] {doc_text.strip()}\n"

        return formatted.strip() if formatted else "No relevant examples found."

    def _get_topic_hint(self, seed_query: str, search_results: Optional[Dict[str, List[Any]]]) -> str:
        """Generates a brief hint about the topic based on query and results."""
        # Basic implementation: use the seed query itself, or potentially summarize keywords from results later.
        # For now, just return the seed query as the primary topic indicator.
        # Could analyze metadata categories if available and consistent.
        # Example: if metadata has 'category' or 'behavior' keys.
        if search_results and search_results.get('metadatas'):
            categories = set()
            behaviors = set()
            for meta in search_results['metadatas']:
                if 'category' in meta: categories.add(str(meta['category']))
                if 'behavior' in meta: behaviors.add(str(meta['behavior'])) # Assuming 'behavior' key exists from JBB data
            if categories or behaviors:
                return f"Topic Hint: {'/'.join(list(categories)[:2]) or '/'.join(list(behaviors)[:2])} (Derived from examples related to '{seed_query[:30]}...')"

        return f"'{seed_query[:50]}...'" # Fallback to using the query itself


    def generate_attack(
        self,
        seed_query: str, # Changed from target_goal
        technique: str = 'generic', # Added technique parameter
        k: Optional[int] = None,
        retrieval_filter: Optional[Dict[str, Any]] = None,
        generation_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a single jailbreak attack prompt for the given seed/topic using RAG
        and a specified technique.

        Args:
            seed_query: A query string or topic description used for retrieving
                        relevant examples.
            technique: The generation technique to use (must match a key in
                       the loaded prompt_templates). Defaults to 'generic'.
            k: Number of examples to retrieve. Overrides default_k if provided.
            retrieval_filter: Optional metadata filter for vector store search.
            generation_options: Optional parameters for the generator LLM.

        Returns:
            A dictionary containing generation results (see previous definition).
            The 'generated_attack_cleaned' is the primary output.
        """
        start_time = time.time()
        k = k if k is not None else self.default_k
        result = {
            "seed_query": seed_query,
            "technique": technique, # Store technique used
            "retrieved_examples": None,
            "topic_hint": None, # Add topic hint field
            "generator_prompt": None,
            "generated_attack_raw": None,
            "generated_attack_cleaned": None,
            "success": False,
            "error": None
        }

        logger.info(f"Generating attack using technique '{technique}' for seed query: '{seed_query[:100]}...'")

        # 1. Retrieve Examples
        search_results = None
        try:
            logger.debug(f"Embedding query for retrieval: '{seed_query[:100]}...'")
            query_embedding = self.query_embedder.encode(seed_query)

            # Optional: Add technique to filter if metadata supports it
            # current_filter = retrieval_filter.copy() if retrieval_filter else {}
            # if self.metadata_has_technique: # Check if technique info exists in DB
            #     current_filter['technique'] = technique
            # logger.info(f"Retrieving {k} examples from vector store. Filter: {current_filter}")
            # search_results = self.retriever.search(query=query_embedding, k=k, filter_criteria=current_filter)

            # Simplified: Use generic filter for now
            logger.info(f"Retrieving {k} examples from vector store. Filter: {retrieval_filter}")
            search_results = self.retriever.search(
                query=query_embedding,
                k=k,
                filter_criteria=retrieval_filter
            )
            result["retrieved_examples"] = search_results
            if search_results:
                 logger.info(f"Retrieved {len(search_results['ids'])} examples.")
            else:
                 logger.warning("Retrieval returned no examples.")

        except Exception as e:
            logger.error(f"Error during retrieval phase: {e}", exc_info=True)
            result["error"] = f"Retrieval failed: {e}"
            # Decide whether to continue or fail
            # formatted_examples = "Error: Retrieval failed." # Allow generation without examples

        # 2. Format examples & Get Topic Hint
        formatted_examples = self._format_retrieved_examples(search_results)
        topic_hint = self._get_topic_hint(seed_query, search_results)
        result["topic_hint"] = topic_hint
        logger.debug(f"Formatted examples for prompt:\n{formatted_examples}")
        logger.debug(f"Topic hint for prompt: {topic_hint}")

        # 3. Select and Construct Generator Prompt
        if technique not in self.prompt_templates:
            logger.error(f"Unknown technique '{technique}'. Available: {list(self.prompt_templates.keys())}")
            result["error"] = f"Unknown technique '{technique}'"
            return result

        template = self.prompt_templates[technique]
        try:
             generator_prompt = template.format(
                 topic_hint=topic_hint, # Use the hint
                 formatted_examples=formatted_examples
             )
             result["generator_prompt"] = generator_prompt
             logger.debug(f"Constructed generator prompt using '{technique}' template (first 300 chars):\n{generator_prompt[:300]}...")
        except KeyError as e:
             logger.error(f"Prompt template error for technique '{technique}': Missing key {e}. Template:\n{template}")
             result["error"] = f"Prompt template error: Missing key {e}"
             return result

        # 4. Call Local LLM
        try:
            logger.info(f"Sending prompt to generator LLM: {self.generator_llm.model_name}")
            llm_output = self.generator_llm.generate(
                prompt=generator_prompt,
                options=generation_options,
                stream=False
            )
            result["generated_attack_raw"] = llm_output
            logger.info(f"Received raw generation (length {len(llm_output)}): '{llm_output[:100]}...'")

            # 5. Clean Output
            cleaned_attack = llm_output.strip()
            # Basic check: Remove potential leading/trailing quotes sometimes added by LLMs
            if len(cleaned_attack) > 1 and cleaned_attack.startswith('"') and cleaned_attack.endswith('"'):
                cleaned_attack = cleaned_attack[1:-1].strip()
            elif len(cleaned_attack) > 1 and cleaned_attack.startswith("'") and cleaned_attack.endswith("'"):
                 cleaned_attack = cleaned_attack[1:-1].strip()

            # Remove common instruction-following artifacts if template asks for "Generated Prompt:"
            if "Generated Prompt:" in cleaned_attack:
                 cleaned_attack = cleaned_attack.split("Generated Prompt:", 1)[-1].strip()


            result["generated_attack_cleaned"] = cleaned_attack
            result["success"] = True # Mark as successful generation

        except Exception as e:
            logger.error(f"Error during generation phase with LLM {self.generator_llm.model_name}: {e}", exc_info=True)
            result["error"] = f"Generation failed: {e}"
            result["success"] = False # Mark as failed generation

        end_time = time.time()
        logger.info(f"Attack generation finished in {end_time - start_time:.2f}s. Generation Success: {result['success']}")
        return result


# --- Example Usage (Modified) ---
if __name__ == "__main__":
    logger.info("--- Attack Generator Example (Technique Focused) ---")

    # --- Configuration ---
    DB_PATH = str(DEFAULT_CHROMA_PATH)
    COLLECTION_NAME = DEFAULT_COLLECTION_NAME
    QUERY_EMBED_MODEL = DEFAULT_EMBEDDING_MODEL
    GENERATOR_LLM_MODEL = DEFAULT_LLM_MODEL
    OLLAMA_URL = DEFAULT_OLLAMA_URL
    TEMPLATES_FILE = DEFAULT_PROMPT_TEMPLATES_PATH
    NUM_EXAMPLES_TO_RETRIEVE = 3

    # Example Seed Query / Topic
    # seed = "illegal activities instructions"
    seed = "How to create malware" # More specific seed for retrieval

    # Techniques to test
    techniques_to_test = ['generic', 'role_play', 'hypothetical']

    # --- Initialization ---
    print("\nInitializing components...")
    # ...(same initialization block as before for query_embedder, vector_store, generator_llm)...
    try:
        query_embedder = EmbeddingModel(model_name=QUERY_EMBED_MODEL)
        vector_store = VectorStore(collection_name=COLLECTION_NAME, persist_directory=DB_PATH)
        if vector_store.collection.count() == 0:
             logger.warning("Vector store is empty! Please run scripts/generate_embeddings.py first.")
        generator_llm = OllamaLLM(model_name=GENERATOR_LLM_MODEL, ollama_url=OLLAMA_URL)

        # Initialize Attack Generator with path to templates
        attack_generator = AttackGenerator(
            retriever=vector_store,
            query_embedder=query_embedder,
            generator_llm=generator_llm,
            prompt_templates_path=TEMPLATES_FILE, # Pass the path
            default_k=NUM_EXAMPLES_TO_RETRIEVE
        )
        print("Components initialized successfully.")
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        print(f"\nError during initialization: {e}.")
        sys.exit(1)


    # --- Generate Attacks for Different Techniques ---
    print(f"\nGenerating attacks for seed query: '{seed}'")

    for tech in techniques_to_test:
        print(f"\n--- Testing Technique: {tech} ---")
        try:
            generation_result = attack_generator.generate_attack(
                seed_query=seed,
                technique=tech,
                generation_options={'temperature': 0.75} # Slightly higher temp for creativity
            )

            print(f"Generation Success: {generation_result['success']}")
            if generation_result['error']: print(f"Error: {generation_result['error']}")

            # Optional: Display retrieved examples
            retrieved = generation_result['retrieved_examples']
            if retrieved and retrieved.get('documents'):
                print(f"Retrieved Examples (Top {len(retrieved['documents'])}):")
                for i, doc in enumerate(retrieved['documents']):
                    print(f"  {i+1}. {doc[:100]}...")
            else:
                print("Retrieved Examples: None")

            print(f"\nGenerated Attack ({tech}):")
            print(generation_result.get('generated_attack_cleaned', "[No attack generated]"))

        except Exception as e:
            logger.error(f"Attack generation failed for technique {tech}: {e}", exc_info=True)
            print(f"Error during attack generation for technique {tech}: {e}")

    logger.info("--- Attack Generator Example End ---")