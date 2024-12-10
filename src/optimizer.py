import os
import json
import logging
from llama_cpp import LogitsProcessorList
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data
from pydantic import BaseModel
from measure_performance import *
from executor import *


class OptimizedCodeSchema(BaseModel):
    """
    Define the JSON schema for the optimized code output.
    """
    original_code: str
    optimized_code: str
    opt: float
    sp: float
    is_correct: bool
    def __init__(self, original_code, optimized_code, opt, sp):
        self.original_code = original_code
        self.optimized_code = optimized_code
        self.opt = opt
        self.sp = sp

    def __lt__ (self, other):
        return [self.sp, self.opt] < [other.sp, self.opt]

    def get_is_correct(self):
        return self.is_correct

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

    def process_batch(self, queries, problem_ids, test_cases_path, mode="single-pass", max_iterations=5):
        """
        Process a batch of queries with either single-pass or iterative refinement.
        """
        for i, query in enumerate(queries):
            logging.info(f"Processing query for Problem ID: {problem_ids[i]}")
            if mode == "single-pass":
                self.single_pass_optimization(query, problem_ids[i], test_cases_path)
            elif mode == "iterative":
                self.iterative_refinement(query, problem_ids[i], test_cases_path, max_iterations)
            else:
                logging.error(f"Invalid mode: {mode}. Please choose 'single-pass' or 'iterative'.")

    def single_pass_optimization(self, query, problem_id, test_cases_path):
        """
        Perform single-pass optimization on a query.
        """
        try:
            logging.info(f"Original Code:\n{query}")
            prompt = self._generate_prompt(query)

            # Generate optimized code
            response = self.llama_model(
                prompt,
                logits_processor=self.logits_processors,
                max_tokens=self.config["max_new_tokens"]
            )
            generated_text = response['choices'][0]['text'].strip()
            parsed_response = json.loads(generated_text)
            optimized_code = parsed_response["optimized_code"]
            

            # Execute and evaluate
            opt, sp = self._execute_and_evaluate(query, optimized_code, problem_id, test_cases_path)

            # Return optimized code
            return OptimizedCodeSchema(original_code=query, optimized_code=optimized_code, opt=opt, sp=sp, is_correct=optimized_code["correct"])
        except Exception as e:
            logging.error(f"Error during single-pass optimization: {e}")

    def iterative_refinement(self, query, problem_id, test_cases_path, max_iterations):
        """
        Perform iterative refinement for optimization.
        """
        current_code = query
        for iteration in range(max_iterations):
            logging.info(f"Starting iteration {iteration + 1} for Problem ID: {problem_id}")
            try:
                prompt = self._generate_prompt(current_code)
                response = self.llama_model(
                    prompt,
                    logits_processor=self.logits_processors,
                    max_tokens=self.config["max_new_tokens"]
                )
                generated_text = response['choices'][0]['text'].strip()
                parsed_response = json.loads(generated_text)
                optimized_code = parsed_response["optimized_code"]

                # Execute and evaluate
                query_results = execute_code_with_test_cases(current_code, problem_id, test_cases_path)
                optimized_results = execute_code_with_test_cases(optimized_code, problem_id, test_cases_path)

                performance = measure_performance(query_results, optimized_results)
                logging.info(f"Iteration {iteration + 1} Performance: {performance}")

                # Check for success criteria
                if performance["OPT"] == 100.0:
                    logging.info(f"Optimization successful after {iteration + 1} iterations.")
                    self._save_result(query, optimized_code)
                    return

                # Update current code for the next iteration
                feedback = self._generate_feedback(query_results, optimized_results)

                current_code = optimized_code + "\n\n# Feedback for next iteration:\n" + feedback

            except Exception as e:
                logging.error(f"Error during iterative refinement in iteration {iteration + 1}: {e}")
                break

        logging.warning(f"Reached maximum iterations for Problem ID: {problem_id}.")
        self._save_result(query, current_code)

    def _generate_prompt(self, code):
        """
        Generate a prompt for LLM to optimize code.
        """
        return (
            f"You are an expert software engineer tasked with optimizing the following code for efficiency.\n"
            f"The optimized code should be functionally equivalent to the original code and execute correctly.\n"
            f"Return only the optimized code without explanation.\n"
            f"Code to optimize:\n{code}\n"
        )

    def _generate_feedback(self, query_results, optimized_results):
        """
        Generate feedback based on execution and performance results.
        """
        feedback = []
        for query_result, optimized_result in zip(query_results, optimized_results):
            if not optimized_result["correct"]:
                feedback.append(
                    f"Test case failed with input: {query_result['input_file']}. "
                    f"Expected output: {query_result['expected_output']}, but got: {optimized_result['result']}."
                )
            elif optimized_result["runtime_ms"] > query_result["runtime_ms"]:
                feedback.append(
                    f"Performance issue: Optimized runtime ({optimized_result['runtime_ms']}ms) "
                    f"is slower than original runtime ({query_result['runtime_ms']}ms)."
                )
        return "\n".join(feedback)

    def _execute_and_evaluate(self, original_code, optimized_code, problem_id, test_cases_path):
        """
        Execute the original and optimized code, then evaluate performance.
        """
        logging.info(f"Executing original code for Problem ID: {problem_id}")
        query_results = execute_code_with_test_cases(original_code, problem_id, test_cases_path)

        logging.info(f"Executing optimized code for Problem ID: {problem_id}")
        optimized_results = execute_code_with_test_cases(optimized_code, problem_id, test_cases_path)

        logging.info("Evaluating performance...")
        performance = measure_performance(query_results, optimized_results)
        logging.info(f"Performance: {performance}")

        self._save_result(original_code, optimized_code)
        return performance["OPT"], performance["SP"]

    def _save_result(self, original_code, optimized_code):
        """
        Save the optimized code to a file.
        """
        result = {"original_code": original_code, "optimized_code": optimized_code}
        with open(self.output_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        logging.info(f"Optimized code saved to {self.output_file}")