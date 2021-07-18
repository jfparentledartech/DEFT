cd src || exit

# train
python train.py tracking,ddd --exp_id pixset_3Dtracking --dataset pixset --pre_hm --shift 0.01 --scale 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --hm_disturb 0.05 --batch_size 4 --gpus 0 --lr 2.5e-4 --save_point 80 --print_iter 20 --num_epochs 2 --val_intervals 1 --eval_val

# train the motion model
python train_prediction.py tracking,ddd --exp_id pixset_3Dtracking_motion_model --dataset pixset --batch_size 2 --gpus 0 --lr 2.5e-4 --num_epochs 100

# test
python test.py tracking,ddd --exp_id pixset_3Dtracking --dataset pixset --load_model ../models/model_pixset.pth --load_model_traj ../models/model_pixset_lstm.pth
