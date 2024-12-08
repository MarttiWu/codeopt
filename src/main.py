import os
import argparse
import logging
import json
from data_loader import create_data_loaders
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import torch

if torch.cuda.is_available():
    device = 0 
elif torch.backends.mps.is_available():
    device = "mps" 
else:
    device = -1

def setup_logging(output_path):
    """
    Set up logging to track the process.
    """
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_path, "execution.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run code optimization with Hugging Face pipeline.")
    parser.add_argument("--mode", type=str, required=True, choices=["optimize"], help="Mode of operation.")
    parser.add_argument("--train_data_path", type=str, required=False, help="Path to training dataset.")
    parser.add_argument("--test_data_path", type=str, required=False, help="Path to testing dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save outputs.")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model name.")
    parser.add_argument("--device", type=str, default="auto", help="Device for model inference (e.g., 'cuda', 'cpu').")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration JSON file.")
    return parser.parse_args()

# Define the schema for the expected output
class OptimizedCodeSchema(BaseModel):
    optimized_code: str

def optimize_code(hf_pipeline, test_loader, config, output_path):
    """
    Optimize code using the Hugging Face pipeline with lm-format-enforcer.
    """
    # Initialize the JSON schema parser
    schema_parser = JsonSchemaParser(OptimizedCodeSchema.model_json_schema())

    # Build the prefix_allowed_tokens_fn for enforcing schema compliance
    prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
        hf_pipeline.tokenizer,
        schema_parser
    )

    for i, batch in enumerate(test_loader):
        if i <= 1:
            continue
        queries = batch["query"]

        for query in queries:
            logging.info(f"Processing query: {query}")

            try:
                prompt = (
                    f"Code to optimize:\n{query}\n"
                    f"You are an expert software engineer tasked with speeding up code for improved efficiency.\n"
                    f"Ensure the optimized code functionality remains unchanged. "
                    f"Respond in the following JSON schema: {OptimizedCodeSchema.model_json_schema()}"
                )

                output_dict = hf_pipeline(
                    prompt,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    max_new_tokens=config.get("max_new_tokens", 200),
                    temperature=config.get("temperature", 0.7),
                    num_return_sequences=1,
                )

                result = output_dict[0]['generated_text'][len(prompt):]
                result = json.loads(result)

                optimized_code = result['optimized_code']

                print("\n--- Original Code ---")
                print(query)
                print("\n--- Optimized Code ---")
                print(optimized_code)

                result = {"original_code": query, "optimized_code": optimized_code}
                output_file = os.path.join(output_path, "optimized_code.jsonl")
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

            except Exception as e:
                logging.error(f"Error during optimization: {e}")
                continue
        break  # For testing purposes, remove this line to process all queries

def main():
    """
    Main function to run the code optimization framework.
    """
    args = parse_arguments()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    setup_logging(args.output_path)
    logging.info("Starting code optimization...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, use_auth_token=True)

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
    except Exception as e:
        logging.error(f"Failed to load the model or tokenizer: {e}")
        exit(1)

    if hf_pipeline.tokenizer.pad_token is None:
        hf_pipeline.tokenizer.add_special_tokens({'pad_token': hf_pipeline.tokenizer.eos_token})
        logging.info("Added pad_token to the tokenizer.")

    logging.info("Creating data loaders...")
    _, test_loader = create_data_loaders(
        train_file=args.train_data_path,
        test_file=args.test_data_path,
        tokenizer=hf_pipeline.tokenizer,
        batch_size=config['batch_size'],
        max_seq_length=config['max_seq_length'],
    )

    logging.info(f"Testing dataset size: {len(test_loader.dataset)}")

    if args.mode == "optimize":
        optimize_code(hf_pipeline, test_loader, config, args.output_path)
    else:
        logging.error("Invalid mode specified.")

if __name__ == "__main__":
    main()