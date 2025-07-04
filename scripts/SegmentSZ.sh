python run_Exp.py \
  --is_training 1 \
  --model_id test \
  --model CausalPD \
  --data train \
  --root_path ./data/SegmentSZ/ \
  --data_path pavement_distress.npy \
  --ext_path ext.csv \
  --seq_len 28 \
  --label_len 14 \
  --pred_len 7 \
  --features M \
  --target OT \
  --freq d \
  --meta_dim 0 \
  --enc_in 7 \
  --d_model 32 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --fc_dropout 0.1 \
  --head_dropout 0.1 \
  --patch_len 16 \
  --stride 4 \
  --padding_patch end \
  --revin 1 \
  --affine 0 \
  --subtract_last 0 \
  --individual 0 \
  --dropout 0.1 \
  --embed timeF \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 50 \
  --patience 5 \
  --loss mse \
  --lradj type3 \
  --use_amp \
  --use_intervention 1 \
  --gpu 0 \
  --des SegmentSZ_exp \
  --d_ff 256 