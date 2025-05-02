# src/rag/embedding.py

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root relative to this file
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd() # Fallback if __file__ is not defined
    logging.warning(f"__file__ not defined. Assuming project root is current working directory: {PROJECT_ROOT}")

# --- Configuration ---
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Good default, relatively small and fast
# Alternative models: "bge-base-en-v1.5", "all-mpnet-base-v2"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "embeddings" / "chroma_db"
DEFAULT_COLLECTION_NAME = "jailbreak_embeddings"

# Ensure the parent directory for ChromaDB exists
DEFAULT_CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Embedding Model Wrapper ---

class EmbeddingModel:
    """
    A wrapper class for loading and using Sentence Transformer models.
    """
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: Optional[str] = None):
        """
        Initializes the EmbeddingModel.

        Args:
            model_name: The name of the sentence-transformer model to load
                        (e.g., 'all-MiniLM-L6-v2').
            device: The device to run the model on ('cpu', 'cuda', etc.).
                    If None, sentence-transformers will automatically choose.
        """
        self.model_name = model_name
        self.device = device
        try:
            logging.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logging.info(f"Model {self.model_name} loaded successfully on device: {self.model.device}")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logging.info(f"Model embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer model '{self.model_name}': {e}")
            raise

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """
        Generates embeddings for the given text(s).

        Args:
            texts: A single string or a list of strings to embed.
            batch_size: The batch size for processing if multiple texts are provided.
            show_progress_bar: Whether to display a progress bar during encoding.

        Returns:
            A numpy array containing the embeddings. If a single string was input,
            the output shape is (1, embedding_dim). If a list of N strings was input,
            the output shape is (N, embedding_dim).
        """
        if not texts:
            return np.array([])

        is_single_string = isinstance(texts, str)
        if is_single_string:
            texts = [texts] # Wrap single string in a list for the model

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True # Ensures NumPy array output
            )
            # The model might return float32, ensure consistency if needed later
            # embeddings = embeddings.astype(np.float32)
            return embeddings
        except Exception as e:
            logging.error(f"Failed to encode texts using model {self.model_name}: {e}")
            # Depending on error, might return partial results or raise
            raise


# --- Vector Store Interface (using ChromaDB) ---

class VectorStore:
    """
    Manages storing and retrieving embeddings using ChromaDB.
    """
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: EmbeddingModel = None, # Use external model by default
        persist_directory: str = str(DEFAULT_CHROMA_PATH),
        # Pass 'embedding_function=None' if providing pre-computed embeddings
        # Or pass a ChromaDB compatible embedding function if you want Chroma to handle it
        embedding_function: Optional[Any] = None
    ):
        """
        Initializes the VectorStore with ChromaDB.

        Args:
            collection_name: Name of the collection within ChromaDB.
            embedding_model: An instance of EmbeddingModel (optional). If provided,
                             it can be used for embedding queries during search.
                             If None, queries must be pre-embedded.
            persist_directory: Path to the directory where ChromaDB should persist data.
            embedding_function: A ChromaDB compatible embedding function. If set
                                (e.g., using chromadb.utils.embedding_functions), ChromaDB
                                can generate embeddings itself. Set to None if adding
                                pre-computed embeddings, which is our primary use case here.
                                For search, if embedding_model is None and embedding_function
                                is None, queries MUST be passed as vectors.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model # Store the model if provided
        self.embedding_function = embedding_function # Store Chroma function if provided

        logging.info(f"Initializing ChromaDB client at: {self.persist_directory}")
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False) # Optional: disable telemetry
            )

            # Get or create the collection
            # If using pre-computed embeddings, the ef should ideally be None or compatible
            # If Chroma generates embeddings, provide a compatible function name or instance
            logging.info(f"Getting or creating ChromaDB collection: {self.collection_name}")
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function # Can be None if adding vectors directly
                # metadata={"hnsw:space": "cosine"} # Optional: Specify distance metric (cosine, l2, ip)
            )
            logging.info(f"Collection '{self.collection_name}' ready. Item count: {self.collection.count()}")

        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB or collection '{self.collection_name}': {e}")
            raise

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Adds embeddings, documents, and metadata to the ChromaDB collection.

        Args:
            embeddings: A numpy array of embeddings (N, embedding_dim).
            documents: A list of N strings (the original text corresponding to each embedding).
            metadatas: A list of N dictionaries containing metadata for each item.
                       Metadata values should be str, int, or float for filtering.
            ids: A list of N unique string IDs for each item. Crucial for updates/deletes.

        Raises:
            ValueError: If the lengths of embeddings, documents, metadatas, and ids do not match.
            Exception: If adding to ChromaDB fails.
        """
        if not (len(embeddings) == len(documents) == len(metadatas) == len(ids)):
            raise ValueError("Lengths of embeddings, documents, metadatas, and ids must match.")
        if len(embeddings) == 0:
            logging.warning("Attempted to add 0 embeddings. Skipping.")
            return

        try:
            # Convert numpy embeddings to list of lists for ChromaDB
            embeddings_list = embeddings.tolist()

            logging.info(f"Adding {len(ids)} items to collection '{self.collection_name}'")
            # Use upsert for idempotency (add or update if ID exists)
            self.collection.upsert(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logging.info(f"Successfully added/updated {len(ids)} items. New count: {self.collection.count()}")

        except Exception as e:
            logging.error(f"Failed to add embeddings to ChromaDB collection '{self.collection_name}': {e}")
            raise

    def search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, List[Any]]]:
        """
        Performs a similarity search in the ChromaDB collection.

        Args:
            query: The query text (string) or query embedding (numpy array, shape (1, dim) or (dim,)).
                   If a string is provided, requires self.embedding_model or self.embedding_function
                   to be configured during initialization to generate the embedding.
            k: The number of nearest neighbors to retrieve.
            filter_criteria: Optional dictionary for filtering results based on metadata
                             (e.g., {'source_dataset': 'advbench'}). See ChromaDB docs
                             for filter syntax (uses $eq, $ne, $in, etc. if needed).

        Returns:
            A dictionary containing the search results ('ids', 'documents', 'metadatas', 'distances'),
            or None if the search fails or returns no results.
            Returns fewer than k results if the collection size (after filtering) is smaller than k.
        """
        if k <= 0:
            logging.warning("Requested k <= 0 neighbors. Returning None.")
            return None

        query_embedding: Optional[List[List[float]]] = None
        query_texts: Optional[List[str]] = None

        if isinstance(query, str):
            if self.embedding_model:
                logging.debug(f"Encoding query text using provided EmbeddingModel: '{query[:50]}...'")
                query_embedding_np = self.embedding_model.encode(query)
                query_embedding = query_embedding_np.tolist() # Needs shape [[...]] for Chroma
            elif self.embedding_function:
                 # Let Chroma handle embedding via its configured function
                 logging.debug(f"Passing query text to ChromaDB embedding function: '{query[:50]}...'")
                 query_texts = [query]
            else:
                logging.error("Cannot encode string query: No EmbeddingModel or Chroma embedding_function provided.")
                raise ValueError("Cannot perform search with string query without an embedding mechanism.")
        elif isinstance(query, np.ndarray):
             logging.debug("Using provided query embedding vector.")
             # Ensure correct shape (1, dim) -> [[...]]
             if query.ndim == 1:
                 query = np.expand_dims(query, axis=0)
             query_embedding = query.tolist()
        else:
             raise TypeError("Query must be a string or a numpy array embedding.")

        try:
            logging.info(f"Performing search in '{self.collection_name}' for {k} neighbors.")
            results = self.collection.query(
                query_embeddings=query_embedding, # Use pre-computed if available
                query_texts=query_texts,          # Use text if letting Chroma embed
                n_results=k,
                where=filter_criteria,             # Apply metadata filter if provided
                include=['metadatas', 'documents', 'distances'] # Specify what to return
            )
            logging.info(f"Search returned {len(results.get('ids', [[]])[0])} results.")

            # Chroma returns results nested in lists, even for a single query. Unpack the first element.
            if results and results.get('ids') and results['ids'][0]:
                 unpacked_results = {
                    'ids': results['ids'][0],
                    'documents': results['documents'][0],
                    'metadatas': results['metadatas'][0],
                    'distances': results['distances'][0]
                 }
                 return unpacked_results
            else:
                 logging.warning("Search returned no results.")
                 return None # Or return empty dict: {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

        except Exception as e:
            logging.error(f"Failed to perform search in ChromaDB collection '{self.collection_name}': {e}")
            # Depending on error, might return None or raise
            return None # Fail gracefully by default


# --- Helper Function ---

def get_text_to_embed(
    record: Dict[str, Any],
    priority_keys: List[str] = ['prompt', 'Goal', 'goal', 'task_description', 'text']
) -> Optional[str]:
    """
    Extracts the most relevant text field from a data record for embedding.

    Iterates through a predefined list of potential keys and returns the value
    of the first key found in the record. Also handles the 'transcript' key
    from Anthropic data by trying to extract the first 'Human:' turn.

    Args:
        record: A dictionary representing a data record.
        priority_keys: A list of keys to check in order of preference.

    Returns:
        The text content found, or None if no relevant key is present or the
        value is empty/invalid.
    """
    for key in priority_keys:
        if key in record and isinstance(record[key], str) and record[key].strip():
            return record[key].strip()

    # Special handling for Anthropic 'transcript' format
    if 'transcript' in record and isinstance(record['transcript'], str):
        transcript = record['transcript'].strip()
        # Try to find the first "Human:" turn as the prompt
        parts = transcript.split('\n\nHuman:')
        if len(parts) > 1:
            first_human_turn = parts[1].split('\n\nAssistant:')[0].strip()
            if first_human_turn:
                return first_human_turn
        # Fallback: Use the whole transcript if no clear first turn found (less ideal)
        # elif transcript:
        #    return transcript

    # Fallback for 'jbb' if it uses a different key not in priority_keys
    # Example: if jbb uses 'Harmful Request'
    # if 'Harmful Request' in record and isinstance(record['Harmful Request'], str) and record['Harmful Request'].strip():
    #     return record['Harmful Request'].strip()

    logging.debug(f"Could not find a suitable text field to embed in record: {record.keys()}")
    return None


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    print("--- Embedding Module Example ---")

    # 1. Initialize Embedding Model
    try:
        embed_model = EmbeddingModel() # Uses default model
    except Exception as e:
        print(f"Failed to initialize Embedding Model: {e}")
        exit()

    # 2. Initialize Vector Store
    # We pass the model instance so the store can embed string queries during search
    try:
        vector_store = VectorStore(
             collection_name="test_collection", # Use a test name
             embedding_model=embed_model,      # Allow string queries
             persist_directory=str(PROJECT_ROOT / "embeddings" / "chroma_test_db") # Use test path
        )
        # Clean up previous test runs if needed (optional)
        # try:
        #      vector_store.client.delete_collection("test_collection")
        #      vector_store = VectorStore("test_collection", embed_model, str(PROJECT_ROOT / "embeddings" / "chroma_test_db"))
        # except: pass # Ignore if collection doesn't exist
    except Exception as e:
        print(f"Failed to initialize Vector Store: {e}")
        exit()


    # 3. Prepare Sample Data
    sample_texts = [
        "How to bake a chocolate cake?",
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "Write a recipe for chocolate cake.", # Similar to first item
        "Who was Albert Einstein?" # Related to third item
    ]
    sample_metadatas = [
        {'source': 'cooking_forum', 'id_original': 'cook1'},
        {'source': 'qa_site', 'id_original': 'qa7'},
        {'source': 'science_wiki', 'id_original': 'sci3'},
        {'source': 'recipe_blog', 'id_original': 'rec4'},
        {'source': 'bio_site', 'id_original': 'bio9'}
    ]
    sample_ids = [f"sample_{i}" for i in range(len(sample_texts))]

    # 4. Generate Embeddings
    print("\nGenerating embeddings for sample data...")
    try:
        sample_embeddings = embed_model.encode(sample_texts)
        print(f"Generated embeddings of shape: {sample_embeddings.shape}")
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        exit()


    # 5. Add to Vector Store
    print("\nAdding embeddings to vector store...")
    try:
        vector_store.add_embeddings(sample_embeddings, sample_texts, sample_metadatas, sample_ids)
    except Exception as e:
        print(f"Failed to add embeddings: {e}")
        exit()


    # 6. Perform Search
    print("\nPerforming similarity search...")
    query_text = "Tell me how to make a cake"
    # query_embedding = embed_model.encode(query_text) # Could also search with embedding directly

    try:
        search_results = vector_store.search(query_text, k=3)

        if search_results:
            print(f"\nSearch results for query: '{query_text}' (Top 3)")
            for i in range(len(search_results['ids'])):
                print(f"  Rank {i+1}:")
                print(f"    ID: {search_results['ids'][i]}")
                print(f"    Distance: {search_results['distances'][i]:.4f}")
                print(f"    Document: {search_results['documents'][i]}")
                print(f"    Metadata: {search_results['metadatas'][i]}")
        else:
            print("Search returned no results.")

    except Exception as e:
        print(f"Search failed: {e}")

    # Example search with filter
    print("\nPerforming search with metadata filter...")
    query_text_sci = "Who invented relativity?"
    try:
         search_results_filtered = vector_store.search(
              query_text_sci,
              k=2,
              filter_criteria={'source': 'science_wiki'} # Only look in science_wiki source
         )
         if search_results_filtered:
             print(f"\nFiltered search results for query: '{query_text_sci}' (source='science_wiki', Top 2)")
             for i in range(len(search_results_filtered['ids'])):
                 print(f"  Rank {i+1}: ID={search_results_filtered['ids'][i]}, Dist={search_results_filtered['distances'][i]:.4f}, Doc='{search_results_filtered['documents'][i]}'")
         else:
             print("Filtered search returned no results.")

    except Exception as e:
        print(f"Filtered search failed: {e}")


    print("\n--- Embedding Module Example End ---")