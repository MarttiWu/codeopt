import ast
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch


# AST-Based Similarity
def ast_similarity(code1, code2):
    def build_ast_graph(code):
        # Ensure the input is a single string
        if isinstance(code, list):  # If it's a list of lines, join them into a single string
            code = "\n".join(code)
        elif not isinstance(code, str):  # If it's not a string or list, raise an error
            raise ValueError(f"Invalid code input: expected string or list, got {type(code)}")

        print("Parsed code:", code)  # Debugging: print the parsed code
        tree = ast.parse(code)
        print("AST tree:", tree)  # Debugging: print the AST tree
        graph = nx.DiGraph()
        print("Graph:", graph)  # Debugging: print the graph

        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                graph.add_edge(type(node).__name__, type(child).__name__)
        return graph

    g1 = build_ast_graph(code1)
    g2 = build_ast_graph(code2)
    try:
        similarity = nx.graph_edit_distance(g1, g2)
        return 1 / (1 + similarity)  # Normalize to similarity score
    except nx.NetworkXError:
        return 0.0


# Semantic Embeddings Similarity
class SemanticSimilarity:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, code):
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    def similarity(self, code1, code2):
        emb1 = self.get_embedding(code1)
        emb2 = self.get_embedding(code2)
        return cosine_similarity(emb1.detach().numpy(), emb2.detach().numpy())[0][0]