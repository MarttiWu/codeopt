import os
import argparse
import logging
import json
# from DS.group.codeopt.src import data_loader
from data_loader import create_data_loaders, create_random_subset
from llama_cpp import Llama
from optimizer import Optimizer
from torch.utils.data import DataLoader


def setup_logging(log_path):
    """
    Set up logging to track the process.
    """
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "execution.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    print(f"Logs will be saved to {log_file}")


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run code optimization with GGUF model.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration JSON file.")
    return parser.parse_args()


def main():
    """
    Main function to run the code optimization framework.
    """
    args = parse_arguments()

    # Load configuration JSON
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Set up logging
    setup_logging(config["log_path"])
    logging.info("Starting code optimization...")

    try:
        # Load GGUF model using llama.cpp
        llama_model = Llama(model_path=config["model_path"], n_threads=4)
    except Exception as e:
        logging.error(f"Failed to load the GGUF model: {e}")
        exit(1)

    logging.info("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        train_file=config["train_data_path"],
        test_file=config["test_data_path"],
        tokenizer=None,  # No tokenizer needed for llama.cpp
        batch_size=config["batch_size"],
        max_seq_length=config["max_seq_length"],
    )

    logging.info(f"Testing dataset size: {len(test_loader.dataset)}")

    subset_size = 2
    test_subset = create_random_subset((test_loader.dataset), subset_size)
    test_loader = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=False)
    logging.info(f"Limited testing dataset size: {len(test_loader.dataset)}")
    
    optimizer = Optimizer(llama_model, config)
    # calculate the test_loader performance, te provess_batch will return the performance  {"OPT": OPT, "SP": SP}
    list_sp = []
    list_opt = []
    for i, batch in enumerate(test_loader):
        # optimizer.process_batch(batch["query"], batch["problem_id"], config["test_cases_path"])
        logging.info("**************************")
        logging.info(f"Processing batch {i}...")
        logging.info("**************************")
        performance = optimizer.process_batch(batch["query"], batch["problem_id"], config["test_cases_path"], mode=config["mode"])
        list_sp.append(performance["SP"])
        list_opt.append(performance["OPT"])
        
    # calculate the average of the SP and OPT
    avg_sp = sum(list_sp) / len(list_sp)
    avg_opt = sum(list_opt) / len(list_opt)
    logging.info(f"Average Speedup Rate: {avg_sp}")
    logging.info(f"Average Percent Optimized: {avg_opt}")
    

if __name__ == "__main__":
    main()