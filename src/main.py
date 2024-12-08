import os
import argparse
import logging
import json
from data_loader import create_data_loaders
from llama_cpp import Llama  # Use llama-cpp-python for GGUF models

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
    parser = argparse.ArgumentParser(description="Run code optimization with GGUF model.")
    parser.add_argument("--mode", type=str, required=True, choices=["optimize"], help="Mode of operation.")
    parser.add_argument("--train_data_path", type=str, required=False, help="Path to training dataset.")
    parser.add_argument("--test_data_path", type=str, required=False, help="Path to testing dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save outputs.")
    parser.add_argument("--model_path", type=str, default="./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf", help="Path to the GGUF model.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration JSON file.")
    return parser.parse_args()

def optimize_code(llama_model, test_loader, config, output_path):
    """
    Optimize code using the GGUF model with llama-cpp-python.
    """
    for i, batch in enumerate(test_loader):
        queries = batch["query"]

        for query in queries:
            logging.info(f"Processing query: {query}")

            try:
                # Generate the prompt
                prompt = (
                    f"You are an expert software engineer tasked with optimizing the following code for efficiency.\n"
                    f"Return only the optimized code without explanation.\n"
                    f"Code to optimize:\n{query}\n"
                    f"Respond with optimized code only."
                )

                # Generate response using llama.cpp
                response = llama_model(prompt, max_tokens=config.get("max_new_tokens", 512))

                print("\n--- Response ---")
                # print(response)

                # Extract the generated text
                generated_text = response['choices'][0]['text']
                optimized_code = generated_text
                # Split the text to isolate the optimized code
                # lines = generated_text.split('\n')
                # optimized_code_lines = [line for line in lines if line.startswith("print(")]  # Adjust logic as needed
                # optimized_code = '\n'.join(optimized_code_lines)

                print("\n--- Original Code ---")
                print(query)
                print("\n--- Optimized Code ---")
                if not optimized_code:
                    print("No optimized code generated.")
                    continue
                print(optimized_code)

                # Save the result
                result = {"original_code": query, "optimized_code": optimized_code}
                output_file = os.path.join(output_path, "optimized_code.jsonl")
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

            except Exception as e:
                logging.error(f"Error during optimization: {e}")
                continue

        # break  # For testing purposes, remove this line to process all queries

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
        # Load GGUF model using llama.cpp
        llama_model = Llama(model_path=args.model_path, n_threads=4)
    except Exception as e:
        logging.error(f"Failed to load the GGUF model: {e}")
        exit(1)

    logging.info("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        train_file=args.train_data_path,
        test_file=args.test_data_path,
        tokenizer=None,  # No tokenizer needed for llama.cpp
        batch_size=config['batch_size'],
        max_seq_length=config['max_seq_length'],
    )

    logging.info(f"Testing dataset size: {len(test_loader.dataset)}")

    if args.mode == "optimize":
        optimize_code(llama_model, train_loader, config, args.output_path)
    else:
        logging.error("Invalid mode specified.")

if __name__ == "__main__":
    main()