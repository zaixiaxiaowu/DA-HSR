gpu_id=0
dataset="food"
seed_list=(42)
inter_file="food_kitchen"
llm_emb_file="itm_emb_np"
sid_emb_file="drq_layer_embs"
model_name="dahsr"

for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --inter_file ${inter_file} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 256 \
                --max_len 50 \
                --lr 1e-4 \
                --dropout_rate 0.2 \
                --gpu_id ${gpu_id} \
                --num_workers 4 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --log \
                --domain "AB" \
                --llm_emb_file ${llm_emb_file} \
                --sid_emb_file ${sid_emb_file} \
                --alpha 0.5 \
                --beta 0.1 \
                --gamma 0.05 \
                --tau1 0.1 \
                --tau2 0.1
done
