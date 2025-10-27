import os
import json
import time
import requests
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
from openai import OpenAI


class LLMJudgeCleaner:
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = config["base_dir"]
        self.candidates = config["candidates"]
        self.context = config.get("context", "qwen2.5-72b-chat")
        self.output_dir = config["output_dir"]

        self.context_window = config.get("context_window", 3)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 90)

        self.provider = config.get("provider", "qwen")
        self.qwen_model = config.get("qwen_model", "qwen2.5-72b-chat")
        self.qwen_api_key = config.get("qwen_api_key") or os.getenv("QWEN_API_KEY")
        self.openrouter_model = config.get("openrouter_model", "qwen/qwen2.5-72b-instruct")
        self.openrouter_api_key = config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")

        # Setup logging
        logging.basicConfig(
            filename=os.path.join(self.output_dir, "llm_judge.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("LLMJudge")
        self.logger.info(f"Initialized LLM Judge with provider: {self.provider}")

        self.openrouter_client = None
        if self.provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key required when using openrouter provider")
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )

        # Book type detection patterns
        self.book_patterns = {
            "log": ["journal", "log", "bord", "voyage", "ship"],
            "meteo": ["meteo", "table", "weather", "climat", "registre"]
        }

    def _extract_page_number(self, filename: str) -> int:
        """Extract page number from complex filenames like 08_OOB_05_001.txt"""
        # Try to find the last numeric part before .txt
        match = re.search(r'(\d+)\.txt$', filename)
        if match:
            return int(match.group(1))

        # Fallback: extract all numbers and use the largest
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(max(numbers, key=lambda x: int(x)))

        # Final fallback: use the whole filename as string
        self.logger.warning(f"Could not extract page number from {filename}, using 0")
        return 0

    def _detect_book_type(self, book_id: str) -> str:
        """Detect book type from its ID"""
        book_id_lower = book_id.lower()
        for book_type, keywords in self.book_patterns.items():
            if any(kw in book_id_lower for kw in keywords):
                return book_type
        return "log"  # Default to log

    def _get_book_instructions(self, book_type: str) -> str:
        """Get specialized instructions for book type"""
        instructions = {
            "log": (
                "You are processing a historical ship logbook. Pay special attention to:\n"
                "- Maritime terminology and ship operations\n"
                "- Consistent date/time formats (e.g., '3ème jour de pluviôse')\n"
                "- Navigation coordinates and weather observations\n"
                "- Sequential event recording\n"
                "Preserve original spelling and formatting."
            ),
            "meteo": (
                "You are processing a historical weather register. Pay special attention to:\n"
                "- Numerical data and measurement units\n"
                "- Tabular structures and alignment\n"
                "- Meteorological symbols and abbreviations\n"
                "- Consistency across daily records\n"
                "Maintain exact numerical values and table formatting."
            )
        }
        return instructions.get(book_type, instructions["log"])

    def _call_qwen_api(self, prompt: str) -> str:
        """Call Qwen API (existing implementation)"""
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.qwen_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]
            },
            "parameters": {
                "result_format": "message",
                "temperature": 0.01,
                "top_p": 0.001,
                "repetition_penalty": 1.05
            }
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    # Fix: Properly handle the response structure
                    output = result.get('output', {})
                    choices = output.get('choices', [])
                    if choices:
                        message = choices[0].get('message', {})
                        content = message.get('content', [])
                        if content and isinstance(content, list):
                            # Find text content in the response
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    return item['text']
                        elif isinstance(content, str):
                            return content
                    self.logger.error(f"Unexpected response format: {result}")
                    return ""
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    try:
                        error_detail = response.json()
                        error_msg = error_detail.get('message', f"HTTP {response.status_code}")
                    except:
                        error_msg = f"HTTP {response.status_code}"
                    self.logger.error(f"API error: {error_msg}")

            except Exception as e:
                self.logger.error(f"Qwen API call failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return ""

    def _call_openrouter_api(self, prompt: str) -> str:
        """Call OpenRouter API"""
        for attempt in range(self.max_retries):
            try:
                response = self.openrouter_client.chat.completions.create(
                    model=self.openrouter_model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=4096,
                    # extra_headers=self.openrouter_headers
                )

                # Add debugging to see the response structure
                self.logger.debug(f"Response object: {response}")

                # Check if response is valid
                if response is None:
                    raise ValueError("Response is None")

                # Check if choices exists and is not empty
                if not hasattr(response, 'choices') or response.choices is None:
                    raise ValueError("Response has no choices attribute or choices is None")

                if len(response.choices) == 0:
                    raise ValueError("Response choices is empty")

                # Check if message content exists
                choice = response.choices[0]
                if not hasattr(choice, 'message') or choice.message is None:
                    raise ValueError("Choice has no message or message is None")

                if not hasattr(choice.message, 'content') or choice.message.content is None:
                    raise ValueError("Message has no content or content is None")

                return choice.message.content.strip()

            except Exception as e:
                self.logger.error(f"OpenRouter API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")

                # Log the full response for debugging if it exists
                if 'response' in locals():
                    self.logger.debug(f"Full response: {response}")

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("All retry attempts exhausted")

        return ""

    def _call_judge_model(self, prompt: str) -> str:
        """Call the judge LLM using configured provider"""
        if self.provider == "qwen":
            return self._call_qwen_api(prompt)
        elif self.provider == "openrouter":
            return self._call_openrouter_api(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _load_context_pages(self, book_id: str, page_files: List[str], current_idx: int) -> Tuple[List[str], List[str]]:
        """Load context pages (2 before and 2 after current page)"""
        prev_pages = []
        next_pages = []

        # Load previous pages (up to 2)
        for i in range(max(0, current_idx - self.context_window), current_idx):
            if i < len(page_files):
                page_id = os.path.splitext(page_files[i])[0]
                # Use the first model's version for context
                context_path = os.path.join(self.base_dir, self.context, book_id, f"{page_id}.txt")
                try:
                    with open(context_path, "r", encoding="utf-8") as f:
                        prev_pages.append(f.read().strip())
                except FileNotFoundError:
                    self.logger.warning(f"Missing context page: {context_path}")

        # Load next pages (up to 2)
        for i in range(current_idx + 1, min(current_idx + 1 + self.context_window, len(page_files))):
            if i < len(page_files):
                page_id = os.path.splitext(page_files[i])[0]
                # Use the first model's version for context
                context_path = os.path.join(self.base_dir, self.context, book_id, f"{page_id}.txt")
                try:
                    with open(context_path, "r", encoding="utf-8") as f:
                        next_pages.append(f.read().strip())
                except FileNotFoundError:
                    self.logger.warning(f"Missing context page: {context_path}")

        return prev_pages, next_pages

    def _build_judge_prompt(
            self,
            candidates: Dict[str, str],
            previous_pages: List[str],
            next_pages: List[str],
            book_type: str
    ) -> str:
        """Construct the prompt for the judge LLM"""
        # Format candidate inputs
        candidates_str = "\n\n".join(
            f"=== Candidate from {model} ===\n{text}"
            for model, text in candidates.items()
        )

        # # Format context
        # context_str = "\n\n".join(
        #     f"--- Previous Page {i+1} ---\n{page}"
        #     for i, page in enumerate(reversed(previous_pages))
        # )

        # Format previous context
        prev_context_str = "\n\n".join(
            f"--- Previous Page {i + 1} ---\n{page}"
            for i, page in enumerate(previous_pages)
        )

        # Format next context
        next_context_str = "\n\n".join(
            f"--- Next Page {i + 1} ---\n{page}"
            for i, page in enumerate(next_pages)
        )

        book_instructions = self._get_book_instructions(book_type)

        return f"""
## Historical Document Transcription Review ##

You are an expert archivist specializing in 18th-19th century maritime documents. 
Your task is to evaluate multiple OCR transcriptions of the same page and produce 
a single authoritative version optimized for historical research and RAG systems.

{book_instructions}

### Context from Previous Pages ###
{prev_context_str if prev_context_str else "No previous pages available"}

### Context from Next Pages ###
{next_context_str if next_context_str else "No next pages available"}


### OCR Candidates for Current Page ###
{candidates_str}

### Evaluation Guidelines ###
1. Accuracy: Select the most faithful representation of the original document
2. Completeness: Prefer versions with less missing text
3. Consistency: Maintain flow with previous pages (dates, terminology)
4. Formatting: Preserve original line breaks, paragraphs, and layout
5. Error Handling: Only correct obvious OCR errors (e.g., 'mαrina' → 'marina')
6. Ambiguity: Mark uncertain readings with [sic?] but preserve original
7. Context Awareness: Use both previous and next pages to resolve ambiguities

### Output Requirements ###
- Return ONLY the final curated text
- Maintain original language (French)
- Preserve historical spellings
- Include all marginal notes in context
- Never add explanatory text
- For tables: maintain alignment with monospace formatting
- If the page is empty or empty content return\"\"

"""

    def _load_page_text(self, book_id: str, page_id: str) -> Dict[str, str]:
        """Load OCR results from all models for a page"""
        page_texts = {}
        for model in self.candidates:
            page_path = os.path.join(self.base_dir, model, book_id, f"{page_id}.txt")
            try:
                with open(page_path, "r", encoding="utf-8") as f:
                    page_texts[model] = f.read().strip()
            except FileNotFoundError:
                self.logger.warning(f"Missing {model}/{book_id}/{page_id}.txt")
                page_texts[model] = ""
        return page_texts

    def process_book(self, book_id: str):
        """Process all pages in a book using LLM judge"""
        book_type = self._detect_book_type(book_id)
        self.logger.info(f"Processing book: {book_id} | Type: {book_type}")

        # Get page list from context model
        model_dir = os.path.join(self.base_dir, self.context, book_id)
        if not os.path.exists(model_dir):
            self.logger.error(f"Book directory not found: {model_dir}")
            return 0

        # Fix: Use proper sorting for complex filenames
        page_files = [f for f in os.listdir(model_dir) if f.endswith(".txt")]
        page_files.sort(key=self._extract_page_number)

        output_dir = os.path.join(self.output_dir, book_id)
        os.makedirs(output_dir, exist_ok=True)

        previous_pages = []
        processed = 0

        for idx, page_file in enumerate(tqdm(page_files, desc=f"Judging {book_id}")):
            page_id = os.path.splitext(page_file)[0]
            output_path = os.path.join(output_dir, f"{page_id}.txt")

            # Load context pages (2 before and 2 after)
            _, next_pages = self._load_context_pages(book_id, page_files, idx)

            # Load candidate OCR results
            candidates = self._load_page_text(book_id, page_id)
            if not any(candidates.values()):
                self.logger.warning(f"Skipping empty page: {page_id}")
                continue

            # Build judge prompt
            prompt = self._build_judge_prompt(candidates, previous_pages, next_pages, book_type)
            self.logger.info(f"Processing {output_path}")
            self.logger.info(f"Prompt\n {prompt}")

            # If exist then skip that file
            curated_text = None
            if not os.path.exists(output_path):
                # Get judge's verdict
                curated_text = self._call_judge_model(prompt)
                self.logger.info(f"Response\n {curated_text}")

                if not curated_text:
                    self.logger.error(f"Failed to get judgment for page {page_id}")
                    continue

                # Save final text
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(curated_text)
            else:
                self.logger.info(f"Skipping existing {output_path}")
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        curated_text = f.read().strip()
                except FileNotFoundError:
                    self.logger.warning(f"Missing existing {output_path}")

            if not curated_text:
                self.logger.error(f"Failed to get judgment for page {page_id}")
                continue
            # Update context window
            previous_pages.append(curated_text)
            if len(previous_pages) > self.context_window:
                previous_pages.pop(0)

            processed += 1

            # Pause to avoid rate limits
            time.sleep(1.5)

        return processed

    def process_library(self, book_ids: List[str] = None):
        """Process multiple books"""
        if book_ids is None:
            model_dir = os.path.join(self.base_dir, self.context)
            book_ids = [d for d in os.listdir(model_dir)
                        if os.path.isdir(os.path.join(model_dir, d))]

        total = 0
        for book_id in book_ids:
            page_count = self.process_book(book_id)
            total += page_count
            self.logger.info(f"Completed {book_id} with {page_count} pages")

        self.logger.info(f"Finished! Total pages curated: {total}")
        return total
