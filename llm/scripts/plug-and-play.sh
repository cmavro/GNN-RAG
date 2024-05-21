SPLIT="test"
DATASET_LIST="RoG-cwq"
BEAM_LIST="3" # "1 2 3 4 5"
MODEL_LIST="llama2-chat-hf"
#PROMPT_LIST="prompts/general_prompt.txt prompts/alpaca.txt prompts/llama2_predict.txt prompts/general_prompt.txt"
PROMPT_LIST="prompts/llama2_predict.txt"

IFS=' '
set -- $PROMPT_LIST

for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        MODEL_NAME=RoG
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
        RULE_PATH_G1=results/gnn/${DATA_NAME}/rearev-sbert/test.info
        RULE_PATH_G2=results/gnn/${DATA_NAME}/rearev-sbert/test.info
        for i in "${!MODEL_LIST[@]}"; do
        
            MODEL_NAME=${MODEL_LIST[$i]}
            PROMPT_PATH=${PROMPT_LIST[$i]}
            
            python src/qa_prediction/predict_answer.py \
                --model_name ${MODEL_NAME} \
                -d ${DATA_NAME} \
                --prompt_path ${PROMPT_PATH} \
                --add_rule \
                --rule_path ${RULE_PATH} \
                --rule_path_g1 ${RULE_PATH_G1} \
                --rule_path_g2 ${RULE_PATH_G2} \
                --predict_path results/KGQA-llms
        done
    done
done