# scripts/run_evaluation.py

import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Ensure the src directory is accessible for imports
import sys
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

SRC_PATH = PROJECT_ROOT / "src"
RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True) # Ensure results directory exists

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Load environment variables BEFORE importing clients
from dotenv import load_dotenv
dotenv_path = PROJECT_ROOT / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    logging.warning(f".env file not found at {dotenv_path}. API keys must be set in environment.")


# Now import from src
try:
    from models.cloud_llm import OpenAIClient, AnthropicClient, DEFAULT_OPENAI_MODEL, DEFAULT_ANTHROPIC_MODEL
    from evaluation.metrics import is_refusal
    from evaluation.llm_judge import LLMJudge, DEFAULT_JUDGE_MODEL
except ImportError as e:
    print(f"Error importing modules. Make sure src is in Python path: {e}")
    print(f"PROJECT_ROOT={PROJECT_ROOT}, SRC_PATH={SRC_PATH}")
    sys.exit(1)

# --- Logging Setup ---
log_file_path = RESULTS_PATH / f"evaluation_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_OUTPUT_FILE = RESULTS_PATH / f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

# --- Helper Functions ---

def load_generated_attacks(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads generated attacks from a JSONL file."""
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"Generated attacks file not found: {path}")
        return []

    attacks = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    attack_data = json.loads(line)
                    # Basic validation: ensure required fields exist
                    if "generated_attack" in attack_data and attack_data["generated_attack"]:
                        attacks.append(attack_data)
                    else:
                         logger.warning(f"Skipping line {i+1} in {path}: Missing or empty 'generated_attack' field.")
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {i+1} in {path}: {e}")
                except Exception as e:
                     logger.warning(f"Error processing line {i+1} in {path}: {e}")
        logger.info(f"Loaded {len(attacks)} valid attack records from {path}.")
        return attacks
    except Exception as e:
        logger.error(f"Failed to read or process generated attacks file {path}: {e}")
        return []

def format_evaluation_result(
    original_attack_data: Dict[str, Any],
    target_model_id: str,
    target_response: Optional[str],
    target_response_error: Optional[str],
    refusal_detected: bool,
    judge_verdict: Optional[str], # COMPLIANCE, REFUSAL, ERROR
    judge_error: Optional[str],
    jailbreak_success: bool,
    run_id: str
    ) -> Dict[str, Any]:
    """Formats the complete evaluation result for saving."""
    return {
        "run_id": run_id,
        "evaluation_id": f"eval_{original_attack_data.get('generation_id', uuid.uuid4().hex[:8])}_{target_model_id.replace(':','-')}", # Combine IDs
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),

        # Info about the attack itself
        "seed_query": original_attack_data.get("seed_query"),
        "technique": original_attack_data.get("technique"),
        "generated_attack": original_attack_data.get("generated_attack"),

        # Info about the target model's response
        "target_model_id": target_model_id,
        "target_response": target_response,
        "target_response_error": target_response_error,

        # Evaluation results
        "evaluation_refusal_detected": refusal_detected,
        "evaluation_judge_verdict": judge_verdict,
        "evaluation_judge_error": judge_error,
        "evaluation_jailbreak_success": jailbreak_success,

        # Optionally include original generation details if needed for analysis
        # "original_generation_success": original_attack_data.get("generation_success"),
        # "original_generation_error": original_attack_data.get("error"),
        # "retrieved_examples_summary": original_attack_data.get("retrieved_examples_summary"),
    }


# --- Main Orchestration Logic ---

def run_evaluation(args):
    """Runs the evaluation process."""
    run_start_time = time.time()
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"--- Starting Evaluation Run ---")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Output will be saved to: {args.output_file}")
    logger.info(f"Logs will be saved to: {log_file_path}")

    # --- Load Attacks ---
    attacks_to_evaluate = load_generated_attacks(args.input_file)
    if not attacks_to_evaluate:
        logger.error(f"No attacks loaded from {args.input_file}. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded {len(attacks_to_evaluate)} attacks to evaluate.")

    # --- Initialize Target Model Clients ---
    target_clients = {}
    if args.target_openai_model:
        try:
            logger.info(f"Initializing OpenAI target client: {args.target_openai_model}")
            # API key should be loaded from env by dotenv
            target_clients["openai"] = OpenAIClient(model_id=args.target_openai_model)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI target client: {e}. Skipping OpenAI targets.")
    if args.target_anthropic_model:
         try:
            logger.info(f"Initializing Anthropic target client: {args.target_anthropic_model}")
            # API key should be loaded from env by dotenv
            target_clients["anthropic"] = AnthropicClient(model_id=args.target_anthropic_model)
         except Exception as e:
            logger.error(f"Failed to initialize Anthropic target client: {e}. Skipping Anthropic targets.")

    if not target_clients:
        logger.error("No target LLM clients initialized successfully. Exiting.")
        sys.exit(1)
    logger.info(f"Initialized target clients for: {list(target_clients.keys())}")


    # --- Initialize Judge ---
    try:
        logger.info(f"Initializing LLM Judge with model: {args.judge_model}")
        # Judge uses OpenAI client internally, key loaded from env
        llm_judge = LLMJudge(judge_model_id=args.judge_model)
    except Exception as e:
        logger.error(f"Failed to initialize LLM Judge: {e}. Exiting.")
        sys.exit(1)
    logger.info("LLM Judge initialized successfully.")

    # --- Evaluation Loop ---
    total_evaluations = len(attacks_to_evaluate) * len(target_clients)
    evaluation_count = 0
    jb_success_count = 0
    refusal_count = 0
    error_count = 0
    logger.info(f"Starting evaluation for {total_evaluations} total evaluations...")

    try:
        with open(args.output_file, 'a', encoding='utf-8') as f_out:
            for i, attack_data in enumerate(attacks_to_evaluate):
                attack_prompt = attack_data.get("generated_attack")
                if not attack_prompt: # Should have been filtered by loader, but double-check
                    logger.warning(f"Skipping attack record {i+1} due to missing prompt.")
                    continue

                for client_key, target_client in target_clients.items():
                    evaluation_count += 1
                    target_model_id = target_client.model_id # Get the specific model ID
                    logger.info(f"--- Evaluating {evaluation_count}/{total_evaluations}: Attack '{attack_data.get('generation_id', i+1)}' vs Target '{target_model_id}' ---")
                    logger.debug(f"Attack Prompt (first 100 chars): {attack_prompt[:100]}...")

                    # Initialize results for this evaluation
                    target_response = None
                    target_response_error = None
                    refusal_detected = True # Default to refusal/fail
                    judge_verdict = None
                    judge_error = None
                    jailbreak_success = False

                    # 1. Get response from target model
                    try:
                        response_dict = target_client.evaluate_prompt(attack_prompt)
                        target_response = response_dict.get("response_text")
                        if not response_dict["success"]:
                            target_response_error = response_dict.get("error", "Target API call failed")
                            logger.warning(f"Target model {target_model_id} failed: {target_response_error}")
                        else:
                             logger.info(f"Target model {target_model_id} responded (Length: {len(target_response or '')})")

                    except Exception as e:
                        target_response_error = f"Script-level exception during target API call: {e}"
                        logger.error(target_response_error, exc_info=True)


                    # 2. Evaluate the response (if obtained successfully)
                    if target_response is not None and target_response_error is None:
                        # Check for explicit refusal patterns first
                        refusal_detected = is_refusal(target_response)

                        if not refusal_detected:
                            # If not an obvious refusal, call the LLM judge
                            logger.info("Response not identified as obvious refusal, invoking LLM judge...")
                            try:
                                judge_result = llm_judge.evaluate_compliance(
                                    attack_prompt=attack_prompt,
                                    response_text=target_response
                                )
                                judge_verdict = judge_result.get("verdict") # COMPLIANCE, REFUSAL, or ERROR
                                judge_error = judge_result.get("error")
                                if judge_error:
                                     logger.warning(f"LLM Judge encountered an error: {judge_error}")
                                # Determine final success based on judge verdict
                                if judge_verdict == "COMPLIANCE":
                                    jailbreak_success = True
                                    jb_success_count += 1
                                else: # Judge verdict was REFUSAL or ERROR
                                     refusal_count +=1 # Count judge-identified refusals/errors here

                            except Exception as e:
                                judge_error = f"Script-level exception during judge call: {e}"
                                logger.error(judge_error, exc_info=True)
                                # Keep jailbreak_success=False, verdict=None, record the error
                        else:
                             # Explicit refusal detected by pattern matching
                             logger.info("Obvious refusal detected by pattern matching.")
                             judge_verdict = "REFUSAL (Pattern)" # Indicate source of refusal detection
                             refusal_count += 1

                    else:
                        # Target model failed to respond correctly
                        logger.warning("Target model did not provide a valid response. Marking as error/failed evaluation.")
                        error_count += 1
                        # Keep refusal_detected=True, jailbreak_success=False


                    # 3. Format and save the result
                    eval_result_data = format_evaluation_result(
                        original_attack_data=attack_data,
                        target_model_id=target_model_id,
                        target_response=target_response,
                        target_response_error=target_response_error,
                        refusal_detected=refusal_detected,
                        judge_verdict=judge_verdict,
                        judge_error=judge_error,
                        jailbreak_success=jailbreak_success,
                        run_id=run_id
                    )
                    f_out.write(json.dumps(eval_result_data) + '\n')

                    # 4. Optional Delay
                    if args.delay > 0 and evaluation_count < total_evaluations:
                        logger.debug(f"Waiting {args.delay}s before next evaluation...")
                        time.sleep(args.delay)

    except IOError as e:
        logger.error(f"Error writing to output file {args.output_file}: {e}")
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user.")
    except Exception as e:
         logger.error(f"An unexpected error occurred during the evaluation loop: {e}", exc_info=True)

    # --- Final Summary ---
    run_end_time = time.time()
    duration = run_end_time - run_start_time
    logger.info("--- Evaluation Run Complete ---")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Total evaluations attempted: {evaluation_count} / {total_evaluations}")
    logger.info(f"Successful jailbreaks: {jb_success_count}")
    logger.info(f"Refusals / Non-compliant: {refusal_count}")
    logger.info(f"Errors (Target or Judge): {error_count}")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info(f"Detailed logs saved to: {log_file_path}")
    logger.info(f"Total time taken: {duration:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated attack prompts against target LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output Args
    parser.add_argument("input_file", type=str, help="Path to the JSONL file containing generated attacks (output from run_generation.py).")
    parser.add_argument("--output-file", type=str, default=str(DEFAULT_OUTPUT_FILE), help="Path to the output JSONL file to save evaluation results.")

    # Target Model Args
    parser.add_argument("--target-openai-model", type=str, default=DEFAULT_OPENAI_MODEL, help="OpenAI model ID to target for evaluation (e.g., gpt-4o-mini). Set empty ('') to disable.")
    parser.add_argument("--target-anthropic-model", type=str, default=DEFAULT_ANTHROPIC_MODEL, help="Anthropic model ID to target for evaluation (e.g., claude-3-haiku-...). Set empty ('') to disable.")
    # Add more target models here if needed (e.g., --target-google-model)

    # Judge Model Args
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="OpenAI model ID to use as the LLM judge (e.g., gpt-4o).")

    # Control Args
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between target API calls (to avoid rate limits). Set 0 to disable.")

    args = parser.parse_args()

    # Basic validation for target models
    if not args.target_openai_model and not args.target_anthropic_model:
         parser.error("At least one target model (--target-openai-model or --target-anthropic-model) must be specified.")

    run_evaluation(args)