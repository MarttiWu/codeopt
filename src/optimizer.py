import os
import json
import logging
from llama_cpp import LogitsProcessorList
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data
from pydantic import BaseModel
from measure_performance import *
from executor import *
from sample_selection.similarity import ast_similarity, SemanticSimilarity
from rank_bm25 import BM25Okapi


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

    def process_batch(self, queries, problem_ids, test_cases_path, mode="single-pass", max_iterations=5, train_loader=None):
        """
        Process a batch of queries with either single-pass or iterative refinement.
        """
        for i, query in enumerate(queries):
            logging.info(f"Processing query for Problem ID: {problem_ids[i]}")
            
            # Retrieve examples using AST-Based and Semantic Embeddings Similarity
            similar_examples = self.retrieve_similar_examples(query, train_loader)

            # Include retrieved examples in the prompt for few-shot learning
            examples_prompt = self.format_examples(similar_examples)
            
            if mode == "single-pass":
                performance  = self.single_pass_optimization(query, problem_ids[i], test_cases_path, examples_prompt)
            elif mode == "iterative":
                performance = self.iterative_refinement(query, problem_ids[i], test_cases_path, max_iterations, examples_prompt)
            else:
                logging.error(f"Invalid mode: {mode}. Please choose 'single-pass' or 'iterative'.")

        return performance
    
    def bm25_preselection(self, train_loader, query_code, top_n=100):
        """
        Preselect top-N similar examples using BM25.
        """
        # Convert train_loader to a list if it is not already a list
        train_data_list = list(train_loader)
        
        train_code_snippets = [example["query"] for example in train_data_list]
        bm25 = BM25Okapi([snippet[0].split() for snippet in train_code_snippets])
        scores = bm25.get_scores(query_code.split())
        
        # Get the top-N indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [train_data_list[i] for i in top_indices]

    def retrieve_similar_examples(self, query, train_loader, top_k=1, bm25_top_n=50):
        """
        Retrieve top-k examples from train_loader using BM25 preselection and AST-Based and Semantic Embeddings Similarity.
        """
        # Step 1: Preselect examples with BM25
        print("Running BM25 preselection...")
        bm25_candidates = self.bm25_preselection(train_loader, query, bm25_top_n)
        print(f"BM25 preselection reduced the search space to {len(bm25_candidates)} examples.")

        # Step 2: Use Semantic Similarity to refine
        retrieved_examples = []
        semantic_sim = SemanticSimilarity()

        for train_data in bm25_candidates:
            train_code = train_data["query"]
            train_target = train_data["target"]

            sem_sim = semantic_sim.similarity(query, train_code)
            print("Semantic similarity:", sem_sim)
            combined_score = sem_sim  # Combine with other metrics if needed

            retrieved_examples.append({
                "example": train_data,
                "combined_score": combined_score
            })

        # Step 3: Sort by combined similarity score
        retrieved_examples = sorted(retrieved_examples, key=lambda x: x["combined_score"], reverse=True)
        return [ex["example"] for ex in retrieved_examples[:top_k]]

    def format_examples(self, examples):
        """
        Format the retrieved examples into a prompt-friendly format.
        """
        prompt = ""
        for example in examples:
            prompt += f"### Example Code:\n{example['query']}\n"
            prompt += f"### Optimized Code:\n{example['target']}\n"
            prompt += "### Diff:\n"
            prompt += "\n".join(example["diff"][0]) + "\n\n"
        return prompt

    def single_pass_optimization(self, query, problem_id, test_cases_path, examples_prompt=""):
        """
        Perform single-pass optimization on a query.
        """
        try:
            logging.info(f"Original Code:\n{query}")
            prompt = self._generate_prompt(query, examples_prompt)

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
            performance = self._execute_and_evaluate(query, optimized_code, problem_id, test_cases_path)
            
            return performance
            
        except Exception as e:
            logging.error(f"Error during single-pass optimization: {e}")

    def _generate_prompt(self, code, examples_prompt):
        """
        Generate a prompt for LLM to optimize code with few-shot examples.
        """
        return (
            f"{examples_prompt}"
            f"You are an expert software engineer tasked with optimizing the following code for efficiency.\n"
            f"The optimized code should be functionally equivalent to the original code and execute correctly.\n"
            f"Return only the optimized code without explanation.\n"
            f"Code to optimize:\n{code}\n"
        )



    def iterative_refinement(self, query, problem_id, test_cases_path, max_iterations, examples_prompt):
        """
        Perform iterative refinement for optimization.
        """
        current_code = query
        best_performance = {"OPT": 0, "SP": 1.0}

        for iteration in range(max_iterations):
            logging.info(f"Starting iteration {iteration + 1} for Problem ID: {problem_id}")
            logging.info(f"Original Code:\n{query}")
            try:
                prompt = self._generate_prompt(current_code, examples_prompt)
                response = self.llama_model(
                    prompt,
                    logits_processor=self.logits_processors,
                    max_tokens=self.config["max_new_tokens"]
                )
                generated_text = response['choices'][0]['text'].strip()
                parsed_response = json.loads(generated_text)
                optimized_code = parsed_response["optimized_code"]
                # log the result
                logging.info(f"iter Optimized Code:\n{optimized_code}")
                
                # Execute and evaluate
                query_results = execute_code_with_test_cases(current_code, problem_id, test_cases_path)
                #log the result
                logging.info(f"iter Query Results: {query_results}")
                
                optimized_results = execute_code_with_test_cases(optimized_code, problem_id, test_cases_path)
                #log the result
                logging.info(f"iter Optimized Results: {optimized_results}")
                
                performance = measure_performance(query_results, optimized_results)
                logging.info(f"Iteration {iteration + 1} Performance: {performance}")
  
                # Check for success criteria
                if performance["OPT"] == 100.0:
                    logging.info(f"Optimization successful after {iteration + 1} iterations.")
                    self._save_result(query, optimized_code)
                    return

                # in order to return te best performance, add up both opt and sp to check if the current performance is better
                if performance["OPT"] + performance["SP"] > best_performance["OPT"] + best_performance["SP"]:
                    best_performance = performance                    
                
                # Update current code for the next iteration
                feedback = self._generate_feedback(query_results, optimized_results)
                current_code = optimized_code + "\n\n# Feedback for next iteration:\n" + feedback

            except Exception as e:
                logging.error(f"Error during iterative refinement in iteration {iteration + 1}: {e}")
                break

        logging.warning(f"Reached maximum iterations for Problem ID: {problem_id}.")
        self._save_result(query, current_code)

        return best_performance

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
        logging.info(f"Query Results: {query_results}")

        logging.info(f"Executing optimized code for Problem ID: {problem_id}")
        optimized_results = execute_code_with_test_cases(optimized_code, problem_id, test_cases_path)
        logging.info(f"Optimized Results: {optimized_results}")
        
        logging.info("Evaluating performance...")
        performance = measure_performance(query_results, optimized_results)
        logging.info(f"Performance: {performance}")

        self._save_result(original_code, optimized_code)
        return performance

    def _save_result(self, original_code, optimized_code):
        """
        Save the optimized code to a file.
        """
        result = {"original_code": original_code, "optimized_code": optimized_code}
        with open(self.output_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        logging.info(f"Optimized code saved to {self.output_file}")