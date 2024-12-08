```bash
conda create -n codeopt python=3.10
```
```bash
conda activate codeopt
```

download the dataset from https://zenodo.org/records/14096664

you have to sign up on hugging face website and obtain token


download mistral-7b-instruct-v0.2.Q5_K_M.gguf
to the models folder
```
mkdir ./models
```
```
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q5_K_M.gguf
--local-dir ./models --local-dir-use-symlinks False
```
