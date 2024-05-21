
SPLIT="test"
DATASET_LIST="RoG-webqsp"
MODEL_NAME=RoG
PROMPT_PATH=prompts/llama2_predict.txt
BEAM_LIST="3" # "1 2 3 4 5"

#GNN-RAG
for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
        RULE_PATH_G1=results/gnn/${DATA_NAME}/rearev-sbert/test.info
        RULE_PATH_G2=None #results/gnn/${DATA_NAME}/rearev-lmsr/test.info

        # no rog
        python src/qa_prediction/predict_answer.py \
            --model_name ${MODEL_NAME} \
            -d ${DATA_NAME} \
            --prompt_path ${PROMPT_PATH} \
            --rule_path ${RULE_PATH} \
            --rule_path_g1 ${RULE_PATH_G1} \
            --rule_path_g2 ${RULE_PATH_G2} \
            --model_path rmanluo/RoG \
            --predict_path results/KGQA-GNN-RAG/rearev-sbert
    done
done


#GNN-RAG-RA
# for DATA_NAME in $DATASET_LIST; do
#     for N_BEAM in $BEAM_LIST; do
#         RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
#         RULE_PATH_G1=results/gnn/${DATA_NAME}/rearev-sbert/test.info
#         RULE_PATH_G2=None #results/gnn/${DATA_NAME}/rearev-lmsr/test.info

#         python src/qa_prediction/predict_answer.py \
#             --model_name ${MODEL_NAME} \
#             -d ${DATA_NAME} \
#             --prompt_path ${PROMPT_PATH} \
#             --add_rule \
#             --rule_path ${RULE_PATH} \
#             --rule_path_g1 ${RULE_PATH_G1} \
#             --rule_path_g2 ${RULE_PATH_G2} \
#             --model_path rmanluo/RoG \
#             --predict_path results/KGQA-GNN-RAG-RA/rearev-sbert
            
#     done
# done