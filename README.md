# The Vulnerability Landscape of Large Language Models: Understanding Jailbreaking Techniques and Their Implications for AI Safety

## 1. Project Overview

This repository contains the code and resources for an undergraduate thesis research project focused on the automated generation and evaluation of jailbreak prompts against safety-aligned Large Language Models (LLMs). The primary goal is to investigate effective automated methods for creating prompts that can bypass the safety mechanisms of modern LLMs, contributing to a better understanding of their vulnerabilities and informing AI safety research.

The core methodology involves:

*   **Retrieval-Augmented Generation (RAG):** Utilizing a curated database of past jailbreak attempts (including their original goals and outcomes) to provide relevant contextual examples to a generator LLM.
*   **Automated Attack Prompt Generation:** Employing a locally hosted LLM (Mistral 7B via Ollama) as a "generator" to analyze these RAG examples and create novel jailbreak prompts based on predefined techniques (e.g., generic, role-playing, hypothetical scenarios).
*   **Systematic Evaluation:** Testing the generated prompts against target cloud-based, safety-aligned LLMs (currently OpenAI GPT-4o Mini and Anthropic Claude 3 Haiku).
*   **Robust Success Assessment:** Evaluating jailbreak success using a combination of pattern-based refusal detection and a powerful LLM-as-judge (GPT-4o) to assess if the target model's response complies with the harmful intent of the generated prompt.

## 2. Key Features & Components

*   **Data Processing Pipeline:** Scripts to standardize various raw jailbreak datasets (Anthropic Red Teaming, AdvBench, Jailbreak Benchmark - `jbb`) into a consistent JSONL format.
*   **Embedding and Vector Store:** Generation of sentence embeddings (`all-MiniLM-L6-v2`) for jailbreak attempts/input prompts from processed datasets, stored in a persistent ChromaDB vector store. Metadata includes the original harmful intent and outcome of each example.
*   **RAG-Powered Attack Generator (`AttackGenerator`):**
    *   Retrieves similar past jailbreak attempts (with goals & outcomes) based on a seed query.
    *   Uses customizable prompt templates (from `configs/prompt_templates.yaml`) for different attack techniques.
    *   Leverages a local LLM (Mistral) to synthesize novel attack prompts.
    *   Supports a Best-of-N strategy for generating multiple candidates.
*   **Comprehensive Evaluation Framework:**
    *   Clients for interacting with target LLMs (OpenAI, Anthropic).
    *   Pattern-based refusal detection (`is_refusal`).
    *   LLM-as-judge (`LLMJudge` using GPT-4o) for nuanced compliance assessment.
*   **Orchestration Scripts:**
    *   `generate_embeddings.py`: Populates the RAG vector store.
    *   `run_generation.py`: Orchestrates RAG and attack prompt generation.
    *   `run_evaluation.py`: Systematically evaluates generated prompts against targets.
*   **Analysis Notebook:** `analyze_evaluation_results.ipynb` for detailed results analysis, visualization, and qualitative review.

## 3. Directory Structure
```
├── configs/ # Configuration files
│ ├── prompt_templates.yaml # Templates for the generator LLM
│ └── seed_queries.txt # Seed queries for generating new attacks
├── data/ # Datasets
│ ├── processed/ # Standardized JSONL datasets (subdirs: anthropic, advbench, jbb)
│ └── raw/ # Original datasets (subdirs: anthropic, advbench, jbb)
├── embeddings/ # Vector store data
│ └── chroma_db/ # ChromaDB persistent storage
├── notebooks/ # Jupyter notebooks for analysis
│ └── analyze_evaluation_results.ipynb
├── results/ # Output files from runs
│ ├── generated_attacks_.jsonl
│ ├── evaluation_results_.jsonl
│ └── log.log # Log files for generation and evaluation runs
├── scripts/ # Orchestration scripts
│ ├── generate_embeddings.py
│ ├── run_generation.py
│ └── run_evaluation.py
├── src/ # Source code modules
│ ├── attacks/
│ │ └── generator.py # AttackGenerator class
│ ├── data/
│ │ └── load.py # Data loading utilities
│ ├── evaluation/
│ │ ├── llm_judge.py # LLMJudge class
│ │ └── metrics.py # Refusal detection logic
│ ├── models/
│ │ ├── cloud_llm.py # Clients for OpenAI/Anthropic APIs
│ │ └── local_llm.py # Client for Ollama (Mistral)
│ └── rag/
│ └── embedding.py # EmbeddingModel and VectorStore (ChromaDB) classes
├── .env # API keys (Git ignored)
├── .gitignore
└── README.md
```


## 4. Setup and Installation

### 4.1. Prerequisites
*   Python 3.8+
*   Git

### 4.2. Installation Steps
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MattiaCervelli/jailbreak-LLM-researchh.git
    cd jailbreak-LLM-researchh
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    (Ensure you have a `requirements.txt` file. If not, create one based on the libraries listed in section 6)
    ```bash
    pip install -r requirements.txt 
    ```
    Key libraries include: `pandas`, `sentence-transformers`, `chromadb`, `requests`, `openai`, `anthropic`, `PyYAML`, `python-dotenv`, `jupyterlab`, `matplotlib`, `seaborn`.

4.  **Set up API Keys:**
    *   Create a file named `.env` in the project root directory.
    *   Add your API keys to this file:
        ```env
        OPENAI_API_KEY="your_openai_api_key_here"
        ANTHROPIC_API_KEY="your_anthropic_api_key_here"
        ```
    *   Replace placeholders with your actual keys. This file is gitignored.

5.  **Set up Ollama and Local LLM (Mistral):**
    *   Install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull the Mistral model (or your desired generator model):
        ```bash
        ollama pull mistral 
        ```
        (Ensure you use the correct model tag, e.g., `mistral:latest` or `mistral:7b`)
    *   Ensure the Ollama server is running (usually starts automatically after installation or can be started with `ollama serve`). The default URL is `http://localhost:11434`.

### 4.3. Prepare Data
1.  **Raw Datasets:** Place your raw datasets (Anthropic Red Teaming, AdvBench, Jailbreak Benchmark/`jbb`) into subdirectories within `data/raw/`.
2.  **Process and Standardize Data:**
    *   You will need to run your (assumed) data processing scripts to convert the raw datasets into the standardized JSONL format expected by `scripts/generate_embeddings.py`.
    *   The standardized files should reside in `data/processed/<dataset_name>/` (e.g., `data/processed/jbb/judge_comparison.jsonl`, `data/processed/advbench/advbench_prompts.jsonl`).
    *   **Important:** Ensure the `jbb` data under `data/processed/jbb/` only contains harmful examples if you want the RAG to focus solely on those. Benign examples should be filtered out before this stage.

## 5. Running the Pipeline

The project follows a multi-step pipeline:

### 5.1. Step 1: Generate Embeddings for RAG
This script processes your standardized datasets, generates embeddings for the jailbreak attempts/input prompts, and stores them along with relevant metadata (original harmful intent, outcome) in a ChromaDB vector store.

*   **Command:**
    ```bash
    python scripts/generate_embeddings.py --force-reload
    ```
    *   `--datasets <name1> <name2>`: Optionally specify which datasets to process (e.g., `jbb anthropic advbench`). If omitted, processes all subdirectories in `data/processed/`.
    *   `--force-reload`: Deletes and recreates the ChromaDB collection. Recommended for the first run or after significant data changes.
    *   Other arguments like `--embedding-model`, `--db-path`, `--collection-name`, `--batch-size` can be used for customization (see script help with `-h`).

*   **Output:** Populates the `embeddings/chroma_db/` directory. Logs are saved in `results/generation_log_*.log`.

### 5.2. Step 2: Generate Attack Prompts
This script uses the RAG database and the local generator LLM (Mistral) to create new jailbreak prompts.

*   **Command:**
    ```bash
    python scripts/run_generation.py \
        --seed-file configs/seed_queries.txt \
        --techniques generic role_play hypothetical \
        --n-candidates 3 \
        -k 3 \
        --output-file results/GenerazioneCompleta.jsonl \
        --generator-llm mistral:latest \
        --delay 0.5 
    ```
    *   `--seed-file`: Path to your seed queries (e.g., `configs/seed_queries.txt`).
    *   `--techniques`: List of attack techniques to use (must match keys in `configs/prompt_templates.yaml`).
    *   `--n-candidates`: Number of attack prompts to generate per seed/technique.
    *   `-k`: Number of RAG examples to retrieve.
    *   `--output-file`: Where to save the generated JSONL attack prompts.
    *   `--generator-llm`: Ollama model to use.
    *   `--delay`: Delay between base generation attempts.
    *   Check `python scripts/run_generation.py -h` for all options.

*   **Output:** A JSONL file (e.g., `results/GenerazioneCompleta.jsonl`) containing the generated attack prompts and associated metadata. Logs are saved in `results/generation_log_*.log`.

### 5.3. Step 3: Evaluate Generated Attacks
This script tests the generated attack prompts against the target cloud LLMs and uses the LLM-as-judge for evaluation.

*   **Command:**
    ```bash
    python scripts/run_evaluation.py \
        results/GenerazioneCompleta.jsonl \
        --output-file results/evaluation_results_GenerazioneCompleta.jsonl \
        --target-openai-model gpt-4o-mini \
        --target-anthropic-model claude-3-haiku-20240307 \
        --judge-model gpt-4o \
        --delay 1
    ```
    *   Positional argument: Path to the generated attacks JSONL file.
    *   `--output-file`: Where to save evaluation results.
    *   `--target-openai-model`, `--target-anthropic-model`: Specify target LLMs. Set to `""` to disable one.
    *   `--judge-model`: LLM to use as the judge.
    *   `--delay`: Delay between target API calls.
    *   Check `python scripts/run_evaluation.py -h` for all options.

*   **Output:** A JSONL file (e.g., `results/evaluation_results_GenerazioneCompleta.jsonl`) with detailed evaluation data for each attack. Logs are saved in `results/evaluation_log_*.log`.

### 5.4. Step 4: Analyze Results
Use the Jupyter Notebook `notebooks/analyze_evaluation_results.ipynb` to load and analyze the final evaluation results.

1.  Open the notebook in VS Code or JupyterLab.
2.  Modify the `results_filename` variable in the second code cell to point to your evaluation output file (e.g., `evaluation_results_GenerazioneCompleta.jsonl`).
3.  Run the cells sequentially to generate statistics, charts, and perform qualitative analysis.

## 6. Key Technologies & Libraries

*   **Python 3**
*   **Core ML/NLP:**
    *   `sentence-transformers`: For generating text embeddings for RAG.
    *   `Ollama`: For running local LLMs (Mistral 7B as generator).
    *   `openai`: Python client for OpenAI API (GPT-4o Mini target, GPT-4o judge).
    *   `anthropic`: Python client for Anthropic API (Claude 3 Haiku target).
*   **Vector Database:**
    *   `chromadb`: For creating and managing the RAG vector store.
*   **Data Handling & Utilities:**
    *   `pandas`: For data analysis in the notebook.
    *   `PyYAML`: For loading prompt templates from `configs/prompt_templates.yaml`.
    *   `python-dotenv`: For managing API keys via `.env` file.
    *   `requests`: For interacting with the Ollama API.
*   **Analysis & Visualization:**
    *   `Jupyter Notebook / JupyterLab`
    *   `matplotlib`
    *   `seaborn`
    *   `numpy`

## 7. Configuration Files

*   `configs/prompt_templates.yaml`: Contains the prompt templates used by the `AttackGenerator` to instruct Mistral for different techniques (generic, role-play, hypothetical). These templates incorporate placeholders for `{topic_hint}` and `{formatted_examples}` (RAG context).
*   `configs/seed_queries.txt`: A text file listing seed queries, categorized by harmful intent groups (e.g., `## Illegal Activities & Hacking`). Used by `run_generation.py` to initiate the attack generation process.

## 8. Latest Results Summary

The latest full run utilized an improved RAG system providing richer context (original goal and outcome of retrieved examples) to the Mistral generator LLM.
*   **Input:** ~100 seed queries, 3 techniques, N=3 candidates per seed/technique (900 generated prompts).
*   **Evaluation:** 900 prompts vs. GPT-4o Mini & Claude 3 Haiku (1800 total evaluations). Judge: GPT-4o.
*   **Overall Jailbreak Success Rate: ~42%**
    *   GPT-4o Mini Success Rate: ~52%
    *   Claude 3 Haiku Success Rate: ~32%

These results indicate an improvement from previous runs, suggesting the refined RAG approach is beneficial.

## 9. Future Work (Potential Directions)

*   Test against a wider range of target LLMs.
*   Experiment with different generator LLMs (local or cloud-based).
*   Explore more sophisticated RAG retrieval strategies (e.g., re-ranking, query transformation).
*   Develop and test new jailbreaking techniques and prompt templates.
*   Investigate the characteristics of generated prompts that are most effective.
*   Analyze the types of refusals and evasions from target LLMs.
*   Explore automated methods for red-teaming the generator LLM itself.
*   Investigate potential defenses against the generated jailbreaks.

## 10. Note on Reproducibility

*   LLM outputs can be stochastic. While seeds are used for the generator LLM and target LLM API calls typically have temperature settings, exact replication of generated text or target responses might vary slightly across runs or environments.
*   API access and potential changes to cloud LLM safety mechanisms can affect results over time.
*   The RAG database content (embeddings) depends on the exact versions of the source datasets and the embedding model used.