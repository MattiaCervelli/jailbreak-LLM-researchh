import logging
import time
import yaml # Import YAML
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

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
            # (Keep this function as it was in your provided code - lines 84-101)
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

        # START OF MODIFIED SECTION for _format_retrieved_examples
    def _format_retrieved_examples(self, search_results: Optional[Dict[str, List[Any]]]) -> str:
            """Formats the retrieved documents and their metadata for inclusion in the prompt."""
            if not search_results or not search_results.get('documents'):
                return "No relevant examples found in the database."

            formatted_output_str = "" 
            for i, doc_text in enumerate(search_results['documents']): 
                meta = search_results['metadatas'][i] if search_results.get('metadatas') and i < len(search_results['metadatas']) else {}
                
                source = meta.get('source_dataset', 'unknown_source')
                original_goal_text = meta.get('original_harmful_intent', meta.get('goal', 'Goal N/A')) # Fallback to 'goal' if new key isn't there
                outcome_text = meta.get('outcome_classification', 'OUTCOME_UNKNOWN')
                
                rating_info = f" (Rating: {meta.get('rating', 'N/A')})" if 'rating' in meta and source == 'anthropic' else ""
                jbb_gpt4_cf_info = f" (gpt4_cf: {meta.get('gpt4_cf', 'N/A')})" if 'gpt4_cf' in meta and source == 'jbb' else ""
                
                formatted_output_str += f"Example {i+1} (Source: {source}):\n"
                formatted_output_str += f"  Original Harmful Goal: {original_goal_text}\n"
                formatted_output_str += f"  Retrieved Jailbreak/Request Text: {doc_text.strip()}\n" 
                formatted_output_str += f"  Outcome of this original attempt: {outcome_text}{rating_info}{jbb_gpt4_cf_info}\n"
                # You can add more specific metadata if useful for a particular source.
                # For instance, for 'jbb', you might want to show 'human_majority' as well.
                # if source == 'jbb' and 'human_majority' in meta:
                #     formatted_output_str += f"    JBB Human Majority: {meta.get('human_majority')}\n"
                formatted_output_str += "---\n"
                
            return formatted_output_str.strip() if formatted_output_str else "No relevant examples found."
        # END OF MODIFIED SECTION

        # START OF MODIFIED SECTION for _get_topic_hint
    def _get_topic_hint(self, seed_query: str, search_results: Optional[Dict[str, List[Any]]]) -> str:
            """Generates a brief hint about the topic based on query and retrieved example intents."""
            if search_results and search_results.get('metadatas'):
                intents_from_results = []
                for meta in search_results['metadatas'][:3]: # Look at top 3 for hints
                    # Use the standardized key, with fallback to 'goal' for older embedded data
                    intent = meta.get('original_harmful_intent', meta.get('goal'))
                    if intent and str(intent) not in intents_from_results: # Ensure unique and string
                        intents_from_results.append(str(intent)[:70] + "...") # Truncate long intents
                
                if intents_from_results:
                    return f"Seed Query: '{seed_query[:50]}...'. Related example intents: [{'; '.join(intents_from_results)}]"
            
            return f"Seed Query: '{seed_query[:50]}...'" # Fallback
        # END OF MODIFIED SECTION

    def generate_attack(
            self,
            seed_query: str, 
            technique: str = 'generic', 
            k: Optional[int] = None,
            retrieval_filter: Optional[Dict[str, Any]] = None,
            generation_options: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            # (Keep this function largely as it was in your provided code - lines 144-269)
            # The main change is that it now calls the modified _format_retrieved_examples and _get_topic_hint
            start_time = time.time()
            k_to_use = k if k is not None else self.default_k # Renamed to avoid conflict with outer k
            result = {
                "seed_query": seed_query,
                "technique": technique,
                "retrieved_examples": None,
                "topic_hint": None, 
                "generator_prompt": None,
                "generated_attack_raw": None,
                "generated_attack_cleaned": None,
                "success": False,
                "error": None
            }
            logger.info(f"Generating attack using technique '{technique}' for seed query: '{seed_query[:100]}...'")

            search_results = None
            try:
                logger.debug(f"Embedding query for retrieval: '{seed_query[:100]}...'")
                query_embedding = self.query_embedder.encode(seed_query)
                logger.info(f"Retrieving {k_to_use} examples from vector store. Filter: {retrieval_filter}")
                search_results = self.retriever.search(
                    query=query_embedding,
                    k=k_to_use,
                    filter_criteria=retrieval_filter
                )
                result["retrieved_examples"] = search_results
                if search_results and search_results.get('ids'): # Check if 'ids' exist and is not empty
                    logger.info(f"Retrieved {len(search_results['ids'])} examples.")
                else:
                    logger.warning("Retrieval returned no valid examples (search_results or ids missing/empty).")
            except Exception as e:
                logger.error(f"Error during retrieval phase: {e}", exc_info=True)
                result["error"] = f"Retrieval failed: {e}"

            formatted_examples = self._format_retrieved_examples(search_results)
            topic_hint = self._get_topic_hint(seed_query, search_results) # Now uses new logic
            result["topic_hint"] = topic_hint
            logger.debug(f"Formatted examples for prompt:\n{formatted_examples}")
            logger.debug(f"Topic hint for prompt: {topic_hint}")

            if technique not in self.prompt_templates:
                logger.error(f"Unknown technique '{technique}'. Available: {list(self.prompt_templates.keys())}")
                result["error"] = f"Unknown technique '{technique}'"
                return result
            template = self.prompt_templates[technique]
            try:
                generator_prompt = template.format(
                    topic_hint=topic_hint, 
                    formatted_examples=formatted_examples
                )
                result["generator_prompt"] = generator_prompt
                logger.debug(f"Constructed generator prompt using '{technique}' template (first 300 chars):\n{generator_prompt[:300]}...")
            except KeyError as e:
                logger.error(f"Prompt template error for technique '{technique}': Missing key {e}. Template:\n{template}")
                result["error"] = f"Prompt template error: Missing key {e}"
                return result
            try:
                logger.info(f"Sending prompt to generator LLM: {self.generator_llm.model_name}")
                llm_output = self.generator_llm.generate(
                    prompt=generator_prompt,
                    options=generation_options,
                    stream=False
                )
                result["generated_attack_raw"] = llm_output
                logger.info(f"Received raw generation (length {len(llm_output)}): '{llm_output[:100]}...'")
                cleaned_attack = llm_output.strip()
                if len(cleaned_attack) > 1 and cleaned_attack.startswith('"') and cleaned_attack.endswith('"'):
                    cleaned_attack = cleaned_attack[1:-1].strip()
                elif len(cleaned_attack) > 1 and cleaned_attack.startswith("'") and cleaned_attack.endswith("'"):
                    cleaned_attack = cleaned_attack[1:-1].strip()
                if "Generated Prompt:" in cleaned_attack: # Common artifact
                    cleaned_attack = cleaned_attack.split("Generated Prompt:", 1)[-1].strip()
                result["generated_attack_cleaned"] = cleaned_attack
                result["success"] = True 
            except Exception as e:
                logger.error(f"Error during generation phase with LLM {self.generator_llm.model_name}: {e}", exc_info=True)
                result["error"] = f"Generation failed: {e}"
                result["success"] = False 
            end_time = time.time()
            logger.info(f"Attack generation finished in {end_time - start_time:.2f}s. Generation Success: {result['success']}")
            return result