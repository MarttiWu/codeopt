import os
import json
import logging
from llama_cpp import LogitsProcessorList
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data
from pydantic import BaseModel


class OptimizedCodeSchema(BaseModel):
    """
    Define the JSON schema for the optimized code output.
    """
    optimized_code: str


class Optimizer:
    def __init__(self, llama_model, config):
        self.llama_model = llama_model
        self.config = config
        self.output_file = os.path.join(config["evaluation_path"], "optimized_code.jsonl")
        os.makedirs(config["evaluation_path"], exist_ok=True)

        # Set up LM Format Enforcer
        tokenizer_data = build_token_enforcer_tokenizer_data(self.llama_model)
        schema_parser = JsonSchemaParser(OptimizedCodeSchema.model_json_schema())
        self.logits_processors = LogitsProcessorList([
            build_llamacpp_logits_processor(tokenizer_data, schema_parser)
        ])

    def process_batch(self, queries):
        for query in queries:
            logging.info(f"Processing query:\n{query}")
            try:
                logging.info(f"Original Code:\n{query}")

                # Generate the prompt
                prompt = (
                    f"You are an expert software engineer tasked with optimizing the following code for efficiency.\n"
                    f"Return only the optimized code without explanation.\n"
                    f"Code to optimize:\n{query}\n"
                )

                # Generate response using llama.cpp
                response = self.llama_model(
                    prompt,
                    logits_processor=self.logits_processors,
                    max_tokens=self.config["max_new_tokens"]
                )

                logging.debug(f"Response:\n{response}")
                generated_text = response['choices'][0]['text'].strip()

                try:
                    parsed_response = json.loads(generated_text)
                    optimized_code = parsed_response["optimized_code"]
                    logging.info(f"Optimized Code:\n{optimized_code}")
                except (json.JSONDecodeError, KeyError) as e:
                    logging.error(f"Error parsing response: {e}")
                    continue

                # Handle empty response
                if not optimized_code:
                    logging.warning("No optimized code generated.")
                    continue

                # Save the result
                self._save_result(query, optimized_code)

            except Exception as e:
                logging.error(f"Error during optimization: {e}")

    def _save_result(self, original_code, optimized_code):
        result = {"original_code": original_code, "optimized_code": optimized_code}
        with open(self.output_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        logging.info(f"Optimized code saved to {self.output_file}")