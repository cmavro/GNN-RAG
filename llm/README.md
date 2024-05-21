## Get Started
We have simple requirements in `requirements.txt`. You can always check if you can run the code immediately.

Please also download `entities_names.json` file from https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp?usp=sharing, as GNNs use the dense graphs. 

## Evaluation
We provide the results of GNN retrieval in `results/gnn`. To evaluate GNN-RAG performance, run `scripts/rag-reasoning.sh`. 

You can also compute perfromance on multi-hop question by `scripts/evaluate_multi_hop.sh`. 

To test different LLMs for KGQA (ChatGPT, LLaMA2), see `scripts/plug-and-play.sh`. 

## Resutls

We append all the results for Table 2: See `results/KGQA-GNN-RAG-RA`. You can look at the actual LLM generations, as well as the KG information retrieved ("input" key) in `predictions.jsonl`.