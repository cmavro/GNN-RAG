This is the code for **GNN-RAG: Graph Neural Retrieval for Large Language Modeling Reasoning**.

The directory is the following:

|----`gnn` folder has the implementation of different KGQA GNNs. 

You can train your own GNNs or you can skip this folder and  use directly the GNN output (retrieved answer nodes) that we computed (`llm/results/gnn`).

|----`llm` folder has the implementation for RAG-based KGQA with LLMs. 

Please see details on how to reproduce results there. 

**Results**: We append all the results for Table 2: See `results/KGQA-GNN-RAG-RA` or `results/KGQA-GNN-RAG`. You can look at the actual LLM generations, as well as the KG information retrieved ("input" key) in predictions.jsonl.