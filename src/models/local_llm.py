import requests
import json
import logging
import time
from typing import Optional, Dict, Any, List, Generator, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_OLLAMA_URL = "http://localhost:11434" # Default Ollama API endpoint
DEFAULT_MODEL = "mistral:latest" # Default model to use if none specified
DEFAULT_TIMEOUT = 120 # Default request timeout in seconds
DEFAULT_KEEPALIVE = "5m" # Default keep_alive setting for Ollama

class OllamaLLM:
    """
    A client to interact with a local LLM served via the Ollama API.
    Handles sending prompts and receiving generated responses.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = DEFAULT_TIMEOUT,
        keep_alive: str = DEFAULT_KEEPALIVE,
        default_options: Optional[Dict[str, Any]] = None,
        default_system_prompt: Optional[str] = None
    ):
        """
        Initializes the OllamaLLM client.

        Args:
            model_name: The name of the Ollama model to use (e.g., 'mistral:latest', 'llama3:8b').
            ollama_url: The base URL of the running Ollama instance.
            timeout: Request timeout in seconds.
            keep_alive: Controls how long the model stays loaded in memory ('5m', '-1' keep indefinitely, '0' unload immediately).
                        See Ollama docs for details.
            default_options: Default generation parameters (e.g., {'temperature': 0.7}).
            default_system_prompt: A default system message to prepend to prompts.
        """
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip('/') # Remove trailing slash if present
        self.generate_endpoint = f"{self.ollama_url}/api/generate"
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.default_options = default_options if default_options is not None else {}
        self.default_system_prompt = default_system_prompt

        logger.info(f"Initialized OllamaLLM client:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  URL: {self.ollama_url}")
        logger.info(f"  Timeout: {self.timeout}s")
        logger.info(f"  Keep Alive: {self.keep_alive}")
        logger.info(f"  Default Options: {self.default_options}")
        logger.info(f"  Default System Prompt: {'Set' if self.default_system_prompt else 'Not Set'}")

        self._check_connection()

    def _check_connection(self):
        """Checks if the Ollama server is reachable."""
        try:
            response = requests.get(self.ollama_url, timeout=5)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            logger.info(f"Successfully connected to Ollama server at {self.ollama_url}")
            # Optional: Check if the specific model is available using /api/tags
            # response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            # if response.ok:
            #     models = response.json().get('models', [])
            #     if not any(m['name'] == self.model_name for m in models):
            #         logger.warning(f"Model '{self.model_name}' not found in Ollama. Ensure it's pulled (`ollama pull {self.model_name}`).")
            # else:
            #      logger.warning(f"Could not retrieve model list from Ollama (Status: {response.status_code}).")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {self.ollama_url}: {e}")
            logger.error("Please ensure Ollama is running and the URL is correct.")
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}") from e

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generates text based on the given prompt using the Ollama API.

        Args:
            prompt: The user prompt for the LLM.
            system_prompt: An optional system prompt to override the default.
                           If None, the default_system_prompt (if set) is used.
                           If explicitly set to '', no system prompt is used.
            options: Optional generation parameters (e.g., temperature) to override defaults.
            stream: If True, returns a generator yielding chunks of the response.
                    If False (default), returns the complete generated text as a single string.

        Returns:
            If stream=False: The complete generated text (str).
            If stream=True: A generator yielding response chunks (str).

        Raises:
            requests.exceptions.RequestException: If the API request fails.
            ValueError: If the response format is unexpected.
            ConnectionError: If connection to Ollama fails.
        """
        request_options = self.default_options.copy()
        if options:
            request_options.update(options)

        final_system_prompt = system_prompt if system_prompt is not None else self.default_system_prompt

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": request_options,
            "keep_alive": self.keep_alive
        }
        if final_system_prompt:
            payload["system"] = final_system_prompt

        logger.debug(f"Sending request to {self.generate_endpoint}")
        logger.debug(f"Payload (prompt truncated): model={payload['model']}, stream={payload['stream']}, system='{str(payload.get('system'))[:50]}...', prompt='{payload['prompt'][:100]}...'")


        try:
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                stream=stream,
                timeout=self.timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            if stream:
                # Return a generator for streaming responses
                return self._process_streaming_response(response)
            else:
                # Process the complete response at once
                response_data = response.json()
                if 'response' in response_data:
                    logger.info(f"Received generation response (length: {len(response_data['response'])}).")
                    # Optionally log context/timing info if needed
                    # logger.debug(f"Eval count: {response_data.get('eval_count')}, Eval duration: {response_data.get('eval_duration')}")
                    return response_data['response'].strip()
                else:
                    logger.error(f"Unexpected response format from Ollama: {response_data}")
                    raise ValueError("Ollama response missing 'response' key.")

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds for model {self.model_name}.")
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s.")
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            # Attempt to get more details from the response body if possible
            error_detail = "No additional detail available."
            try:
                 if not stream and e.response is not None:
                      error_detail = e.response.text
                 logger.error(f"Error detail: {error_detail}")
            except Exception:
                 pass # Ignore errors trying to get error details
            raise ConnectionError(f"Ollama API request failed: {e}. Detail: {error_detail}") from e

    def _process_streaming_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Helper generator to process streaming responses from Ollama."""
        full_response_content = "" # Keep track for logging length at end
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        content = chunk.get('response', '')
                        if content:
                            yield content
                            full_response_content += content
                        # Check if generation is done (last chunk)
                        if chunk.get('done', False):
                            # Optionally log context/timing from the final chunk
                            # eval_count = chunk.get('eval_count')
                            # eval_duration = chunk.get('eval_duration')
                            # logger.debug(f"Stream ended. Eval count: {eval_count}, Eval duration: {eval_duration}")
                            break # Exit the loop once done is true
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON line in stream: {line}")
                    except Exception as e:
                        logger.error(f"Error processing stream chunk: {e}")
                        # Decide whether to continue or break here
                        break
            logger.info(f"Finished processing stream (Total length: {len(full_response_content)}).")
        except Exception as e:
             logger.error(f"Error reading stream from Ollama: {e}")
             raise # Re-raise the error