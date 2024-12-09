import os
import json
import logging
from llama_cpp import LogitsProcessorList
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data
from pydantic import BaseModel
from scipy import optimize
from evaluate import *
from exe_code import *

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
        self.speedup = 0
        os.makedirs(config["evaluation_path"], exist_ok=True)

        # Set up LM Format Enforcer
        tokenizer_data = build_token_enforcer_tokenizer_data(self.llama_model)
        schema_parser = JsonSchemaParser(OptimizedCodeSchema.model_json_schema())
        self.logits_processors = LogitsProcessorList([
            build_llamacpp_logits_processor(tokenizer_data, schema_parser)
        ])

    def process_batch(self, queries, problem_ids, test_cases_path):
        for i,query in enumerate(queries):
            logging.info(f"Processing query:\n{query}")
            try:
                logging.info(f"Original Code:\n{query}")

                # Generate the prompt
                prompt = (
                    f"You are an expert software engineer tasked with optimizing the following code for efficiency.\n"
                    f"The optimized code should be functionally equivalent to the original code and can execute correctly.\n"
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
                    # code_id = find_matching_code(self.config["test_path"], query)
                    # original_result, original_runtime, original_memory = execute_code(query)
                    logging.info("Problem ID: "+str(problem_ids[i]))
                    logging.info("Executing original code...")
                    query_results = execute_code_with_test_cases(query, problem_ids[i], test_cases_path)
                    logging.info("Executing optimized code...")
                    optimized_results = execute_code_with_test_cases(optimized_code, problem_ids[i], test_cases_path)
                    
                    logging.info("query_results: "+str(query_results))
                    logging.info("optimized_results: "+str(optimized_results))

                    # evaluate error and calculate speed up rate

                    logging.info("Evaluating query performance...")
                    query_performance = measure_performance(query_results)
                    optimized_performance = measure_performance(optimized_results)
                    logging.info(f"Original Code Performance: {query_performance}")
                    logging.info(f"Optimized Code Performance: {optimized_performance}") 
                                      
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