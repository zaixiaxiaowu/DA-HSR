CUDA_VISIBLE_DEVICES='0' python RQ-VAE/generate_indices.py\
    --dataset Food_Kitchen \
    --data_path data/food/handled/itm_emb_np_all.pkl \
    --root_path data/food/handled/sid/ \
    --checkpoint best_loss_model.pth \
    --e_dim 128 \
    --num_emb_list 32 256 256 256 256 \
    --layers 512 128