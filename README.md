This is the code for **GNN-RAG: Graph Neural Retrieval for Large Language Modeling Reasoning**.

Paper link: https://arxiv.org/abs/2405.20139

![alt GNN-RAG: The GNN reasons over a dense subgraph to retrieve candidate answers, along
with the corresponding reasoning paths (shortest paths from question entities to answers). The
retrieved reasoning paths -optionally combined with retrieval augmentation (RA)- are verbalized
and given to the LLM for RAG](GNN-RAG.png "GNN-RAG")

The directory is the following:

|----`gnn` folder has the implementation of different KGQA GNNs. 

You can train your own GNNs or you can skip this folder and  use directly the GNN output (retrieved answer nodes) that we computed (`llm/results/gnn`).

|----`llm` folder has the implementation for RAG-based KGQA with LLMs. 

Please see details on how to reproduce results there. 

**Results**: We append all the results for Table 2: See `results/KGQA-GNN-RAG-RA` or `results/KGQA-GNN-RAG`. You can look at the actual LLM generations, as well as the KG information retrieved ("input" key) in predictions.jsonl.


If you find our work and code useful, please cite:
```
@misc{mavromatis2024gnnrag,
      title={GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning}, 
      author={Costas Mavromatis and George Karypis},
      year={2024},
      eprint={2405.20139},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```