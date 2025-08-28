import base64, os, json, time
from typing import List, Union
from io import BytesIO
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datasets import load_dataset
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from tqdm import tqdm

import llms
from schemas.kocem import LocaleType
from utils.data import save_json
from utils.ds import call_features
from utils.eval import evaluate, evaluate_difficulties, parse_multi_choice_response, parse_open_response
from utils.logs import set_logger
# from llms import llm_models

from .prompt import PromptManager


logger = set_logger(__name__)

DS_PATH = os.getenv("DS_PATH")
DS_CACHE_PATH = os.getenv("DS_CACHE_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")



class APIBase:
    ds_path = DS_PATH
    ds_cache_path = DS_CACHE_PATH
    output_path = OUTPUT_PATH

    def __init__(self, 
        locale: LocaleType = "en",
        task: str = "mcqa",
        prompt: str = "mcqa",
        prompt_version: str = "latest",
        **kwargs
    ):
        self.locale = locale
        self.task = task
        self.prompt_name = prompt
        self.prompt = PromptManager(name=prompt, locale=locale, version=prompt_version)
    
    def _set_options(self, sample):
        if self.subset == "Standard_Nomenclature":
            options_raw = sample['options']
        else:
            options_raw = sample['{}_options'.format(self.locale)]
        
        if isinstance(options_raw, list):
            option_list = options_raw
        else:
            option_list = json.loads(options_raw)
        
        return {chr(65 + i): opt for i, opt in enumerate(option_list)}

    @staticmethod
    def _set_image(sample) -> tuple:
        image_raw = sample['image']
        if isinstance(image_raw, dict):
            return image_raw["path"], image_raw["bytes"]
        elif isinstance(image_raw, str):
            image_dict = json.loads(image_raw)
            return image_dict["path"], image_dict["bytes"]
        else:
            raise ValueError("Image data must be a dict or JSON string.")

    def _construct_mcqa_data(self, sample) -> dict:
        if self.subset == "Standard_Nomenclature":
            question = sample['question']
            options = self._set_options(sample)
            answer = sample['answer']
            answer_key = sample['answer_key']
            explanation = sample.get('explanation', '')
            image_path, image_bytes = self._set_image(sample)
        else:
            question = sample['{}_question'.format(self.locale)]
            options = self._set_options(sample)
            answer = sample['{}_answer'.format(self.locale)]
            answer_key = sample['answer_key']
            explanation = sample['{}_explanation'.format(self.locale)]
            image_path, image_bytes = self._set_image(sample)

        return {
            "id": sample['id'],
            "question_type": sample['question_type'],
            "difficulty": sample.get('difficulty', None),
            "human_acc": sample.get('human_acc', None),
            "question": question,
            "options": options,
            "ground_truth": {
                "answer": {answer_key: answer},
                "explanation": explanation
            },
            "image_path": image_path,
            "image_bytes": image_bytes
        }
    
    def construct_data(self, sample: dict) -> dict:
        """
        Construct data for the model from a sample.
        
        Args:
            sample (dict): A single sample from the dataset.
        
        Returns:
            dict: Processed data ready for model inference.
        """
        if self.task == "mcqa":
            return self._construct_mcqa_data(sample)
        raise NotImplementedError(f"Task {self.prompt.task} is not implemented in APIBase.")

    def construct_prompt(self,
            question, 
            options, 
            image_bytes=None
        ) -> List[Union[SystemMessage, HumanMessage]]:
        def _supports_image_input() -> bool:
            try:
                target_full = str(self.model_id)
                target_last = target_full if "grok" in target_full else target_full.split("/")[-1]
                targets = {target_full.lower(), target_last.lower()}
                for fam in llms.llm_models:
                    for lm in getattr(fam, "models", []):
                        names = set()
                        n0 = getattr(lm, "name", "")
                        if n0:
                            names.add(n0)
                        ver = getattr(lm, "version", None)
                        if ver:
                            s0 = getattr(ver, "stable", "")
                            if s0:
                                names.add(s0)
                            for r in getattr(ver, "releases", []) or []:
                                names.add(r)
                        # Compare against full and last segments (case-insensitive)
                        matchable = set()
                        for nm in names:
                            if not nm:
                                continue
                            matchable.add(nm.lower())
                            matchable.add(nm.split("/")[-1].lower())
                        if targets & matchable:
                            modality = getattr(lm, "modality", None)
                            inputs = getattr(modality, "input_type", []) if modality else []
                            return "image" in [str(x).lower() for x in inputs]
            except Exception:
                pass
            return False
        def _detect_image_mime(b: bytes) -> str:
            try:
                if b.startswith(b"\xFF\xD8\xFF"):
                    return "image/jpeg"
                if b.startswith(b"\x89PNG\r\n\x1a\n"):
                    return "image/png"
                if b.startswith(b"GIF8"):
                    return "image/gif"
                if b.startswith(b"RIFF") and b[8:12] == b"WEBP":
                    return "image/webp"
            except Exception:
                pass
            return "application/octet-stream"
        
        # Build textual portion
        text_prompt = self.prompt.human.format(
            question=question,
            options="\n".join(f"({key}) {value}" for key, value in options.items())
        )
        logger.debug(f"Prompt: {text_prompt}")

        if image_bytes and image_bytes != "null" and _supports_image_input():
            mime = _detect_image_mime(image_bytes)
            b64 = base64.b64encode(image_bytes).decode()
            # OpenAI-compatible multi-modal content structure
            content = [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        elif image_bytes and image_bytes != "null":
            logger.debug("Skipping image content: model does not support image input modality.")
            content = text_prompt
        else:
            content = text_prompt

        if "gpt-5" in self.model_id:
            return [
                {"role": "system", "content": self.prompt.system},
                {"role": "user", "content": content}
            ]
        else:
            return [
                SystemMessage(content=self.prompt.system),
                HumanMessage(content=content)
            ]
    
    # def _load_dataset(self, subset: str, split: str):
    #     """Load a dataset split with safe fallbacks to handle empty/invalid splits.
    # 
    #     Order of attempts:
    #     1) Normal cached load
    #     2) Force redownload with verification disabled
    #     3) Streaming mode (no schema casting)
    #     """
    # 
    #     try:
    #         try:
    #             return load_dataset(
    #                 path=DS_PATH, 
    #                 name=subset, 
    #                 split=split, 
    #                 cache_dir=DS_CACHE_PATH,
    #                 verification_mode="no_checks",
    #                 features=call_features(subset),
    #             )
    #         except TypeError:
    #             # Older datasets versions don't support verification_mode
    #             return load_dataset(
    #                 path=DS_PATH, 
    #                 name=subset, 
    #                 split=split, 
    #                 cache_dir=DS_CACHE_PATH,
    #                 ignore_verifications=True,
    #                 features=call_features(subset)
    #             )
    #     except Exception as e:
    #         logger.error(f"Failed to load dataset {DS_PATH} for subset {subset} and split {split}: {e}")
    #         try:
    #             logger.info("Retrying with force_redownload and no_checks...")
    #             try:
    #                 return load_dataset(
    #                     path=DS_PATH,
    #                     name=subset,
    #                     split=split,
    #                     cache_dir=DS_CACHE_PATH,
    #                     download_mode="force_redownload",
    #                     verification_mode="no_checks",
    #                     features=call_features(subset)
    #                 )
    #             except TypeError:
    #                 return load_dataset(
    #                     path=DS_PATH,
    #                     name=subset,
    #                     split=split,
    #                     cache_dir=DS_CACHE_PATH,
    #                     download_mode="force_redownload",
    #                     ignore_verifications=True,
    #                     features=call_features(subset)
    #                 )
    #         except Exception as e2:
    #             logger.warning(f"Retrying with streaming mode due to persistent load errors: {e2}")
    #             return load_dataset(
    #                 path=DS_PATH,
    #                 name=subset,
    #                 split=split,
    #                 streaming=True,
    #                 features=call_features(subset)
    #             )

    def _parse_response(self, handler: dict) -> dict:
        """
        Parse the model's response based on the task type.
        
        Args:
            handler (dict): The response from the model.
        
        Returns:
            dict: Parsed response with the model's answer.
        """
        if handler['question_type'] == 'multiple-choice':
            return parse_multi_choice_response(
                handler["model_answer"],
                handler["question"],
                handler["options"].keys(),
                handler["options"]
            )
        elif self.task == "open_response":
            return parse_open_response(handler["model_answer"])
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented in APIBase.")

    def __call__(self,
        subset: str,
        split: str,
        max_retries: int = 5,
        max_timeout: int = 60,
        calculate_difficulty: bool = False,
        override: bool = False,
        reduce_image_on_retry: bool = True,
        reduce_scale: float = 0.5,
    ) -> dict[str, str]:
        """
        Run MCQA inference over configured splits/subsets.

        Args:
            subset (str): The name of the dataset subset to process.
            split (str): The dataset split to process (e.g., 'train', 'dev', 'test').
            max_retries (int): Maximum number of retries for model invocation.
            max_timeout (int): Maximum timeout in seconds for each model invocation.

        Returns:
            dict[str, str]: Paths to the output file and result file.
        """
        # Load dataset split
        self.subset = subset
        self.split = split

        # Prepare output directory and save results
        output_dir = os.path.join(self.output_path, self.prompt_name, self.locale, self.model_id.split('/')[-1], split, subset)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "output.json")
        result_file = os.path.join(output_dir, "result.json")
        evaluatation_file = os.path.join(output_dir, "evaluation.json")
        if not override and os.path.exists(result_file) and os.path.exists(evaluatation_file):
            logger.info(f"Skipping {subset} - {split} as result files already exist and override is False.")
            return {
                "evaluation": evaluatation_file, 
                "output": output_file, 
                "result": result_file
            }

        # dataset = self._load_dataset(subset, split)
        dataset = load_dataset(
            path=self.ds_path, 
            name=subset, 
            split=split, 
            cache_dir=self.ds_cache_path,
            verification_mode="no_checks",
            features=call_features(subset),
        )

        results = []
        for sample in tqdm(dataset, desc=f"Processing: {subset}/{split}"):
            handler = self.construct_data(sample)
            logger.debug(f"Sample ID: {handler['id']}")
            image_bytes = handler.pop('image_bytes', None)

            def _shrink_image_bytes(b: bytes, scale: float = 0.5, target_max_bytes: int | None = None) -> bytes:
                try:
                    with Image.open(BytesIO(b)) as img:
                        img = img.convert("RGB")
                        w, h = img.size
                        # Start with initial scale
                        cur_scale = max(0.05, min(1.0, scale))
                        qualities = [85, 75, 65, 55, 45]
                        for _ in range(6):  # up to 6 scale steps
                            nw = max(8, int(w * cur_scale))
                            nh = max(8, int(h * cur_scale))
                            resized = img.resize((nw, nh), Image.LANCZOS)
                            # Try multiple JPEG qualities
                            for q in qualities:
                                out = BytesIO()
                                resized.save(out, format="JPEG", quality=q, optimize=True)
                                data = out.getvalue()
                                if target_max_bytes is None or len(data) <= target_max_bytes:
                                    return data
                            # If still too big, reduce scale further
                            cur_scale *= 0.8
                        # Fallback to final attempt PNG at smallest size
                        out = BytesIO()
                        resized.save(out, format="PNG", optimize=True)
                        return out.getvalue()
                except Exception as e:
                    logger.warning(f"Image downscale failed; using original. Reason: {e}")
                    return b

            response = None
            if reduce_image_on_retry and image_bytes and max_retries >= 2:
                # Manual control: first attempt original image, on first failure retry with downscaled image.
                attempts_left = max_retries
                current_bytes = image_bytes
                attempt_idx = 1
                while attempts_left > 0:
                    prompt_msgs = self.construct_prompt(
                        question=handler['question'],
                        options=handler['options'],
                        image_bytes=current_bytes
                    )
                    try:
                        # Delegate per-attempt execution to model, with inner retries disabled
                        response = self._invoke_with_retry(
                            prompt_msgs=prompt_msgs,
                            max_retries=1,
                            max_timeout=max_timeout,
                        )
                        break
                    except Exception as e:
                        attempts_left -= 1
                        if attempt_idx == 1 and attempts_left > 0:
                            # Prepare reduced image and retry once
                            logger.warning("First attempt failed; retrying once with downscaled image bytes.")
                            # If provider is Anthropic/Claude, respect ~5MB cap
                            limit = None
                            mid_lower = str(self.model_id).lower()
                            if ("anthropic" in mid_lower) or ("claude" in mid_lower):
                                limit = 5 * 1024 * 1024
                            current_bytes = _shrink_image_bytes(image_bytes, scale=reduce_scale, target_max_bytes=limit)
                            attempt_idx += 1
                            continue
                        # No special handling left; re-raise on last failure
                        if attempts_left == 0:
                            raise
                        attempt_idx += 1
                if response is None:
                    # Shouldn't happen; safeguard
                    raise RuntimeError("Model invocation failed with image-retry logic.")
            else:
                # Original behavior
                prompt_msgs = self.construct_prompt(
                    question=handler['question'],
                    options=handler['options'],
                    image_bytes=image_bytes
                )
                response = self._invoke_with_retry(
                    prompt_msgs=prompt_msgs,
                    max_retries=max_retries,
                    max_timeout=max_timeout
                )
            answer = str(getattr(response, "content", str(response))).strip()
            logger.debug(f"Model response: {answer}")

            handler['model_answer'] = answer
            handler["parsed_pred"] = self._parse_response(handler)
            handler["full_response"] = getattr(response, 'dict', lambda: str(response))()
            
            results.append(handler)

        logger.debug(f"Saving temporal results to {output_file}")
        save_json(output_file, results)

        judge_dict, metric_dict = evaluate(results)
        for result in results:
            result.update({"judge": judge_dict[result['id']]["judge"]})

        save_json(evaluatation_file, judge_dict)
        logger.debug(f"Evaluation file saved at {evaluatation_file}.")
        
        save_json(output_file, results)
        logger.debug(f"Output file saved at {output_file}.")

        if calculate_difficulty:
            metric_dict.update({"difficulties": evaluate_difficulties(judge_dict)})

        save_json(result_file, metric_dict)
        logger.debug(f"Result file saved at {result_file}.")
        
        # Log and returns the output file path
        return {
            "evaluation": evaluatation_file, 
            "output": output_file, 
            "result": result_file
        }

    # ---------------- Internal helpers -----------------
    def _invoke_with_retry(self, prompt_msgs, max_retries: int, max_timeout: int):
        """Invoke the model with retry and timeout logic.

        Args:
            prompt_msgs: Messages to send to model.
            max_retries (int): Total attempts before failing.
            max_timeout (int): Per-attempt timeout in seconds (0 / negative disables).
        """

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                if max_timeout and max_timeout > 0:
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(self.model.invoke, prompt_msgs)
                        return fut.result(timeout=max_timeout)
                else:
                    return self.model.invoke(prompt_msgs)
            except (FuturesTimeout, Exception) as e:  # broad catch to retry transient issues
                last_err = e
                if attempt == max_retries:
                    logger.error(f"Model invoke failed after {attempt} attempts: {e}")
                    raise
                backoff = min(2 ** (attempt - 1), 10)
                logger.warning(f"Invoke error (attempt {attempt}/{max_retries}): {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
        # Should not reach here
        raise last_err if last_err else RuntimeError("Unknown invocation failure")

    @property
    def model_id(self):
        """
        Get the model ID for the APIBase instance.
        
        Returns:
            str: The model ID (i.g. openai/gpt-4.1).
        """
        return self._model_id if hasattr(self, '_model_id') else NotImplemented
    
    @model_id.setter
    def model_id(self, value: str):
        """
        Set the model ID for the APIBase instance.
        
        Args:
            value (str): The model ID to set.
        """
        self._model_id = value

    @property
    def model(self):
        """
        Get the initialized chat model for the APIBase instance.
        
        Returns:
            The initialized chat model.
        """
        return self._model if hasattr(self, '_model') else NotImplemented
    
    @model.setter
    def model(self, value: Runnable):
        """
        Set the chat model for the APIBase instance.
        
        Args:
            value (Runnable): The chat model to set.
        """
        self._model = value



__all__ = ["APIBase"]