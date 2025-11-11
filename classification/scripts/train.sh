# If lbm kernel works well but there is an issue with "Could not find 
# selective_scan_interface_lbm", try replace .. with the absolute path,
# or create a symbolic link by "ln -s ./lbm ./classification".
# lbvim tiny
PYTHONPATH=.:..:$PYTHONPATH CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py --model lbvim_tiny_patch16_224 --batch-size 512 --drop-path 0.05 --weight-decay 0.05 --lr 1e-3 --num_workers 12 --data-path /dev/shm/jzhang/imagenet1k --output_dir /home/jingwezhang1/result/bivim/vim_IN1k_para/bivim_layerflip --no_amp &> /home/jingwezhang1/result/bivim/vim_IN1k_para/bivim_layerflip/log.txt

# lbvim 300
PYTHONPATH=.:..:$PYTHONPATH torchrun --standalone --nnodes=1 --nproc-per-node=8 main.py --model lbvim_300_patch16_224 --batch-size 128 --drop-path 0.05 --weight-decay 0.05 --lr 1e-3 --num_workers 8 --data-path /scratch/KurcGroup/jingwei/data/ImageNet --output_dir  /scratch/KurcGroup/jingwei/result/bivim/vim_IN1k_para/bivim_300_layerflip --no_amp &> /scratch/KurcGroup/jingwei/result/bivim/vim_IN1k_para/bivim_300_layerflip/log.txt

# lbvim small
PYTHONPATH=.:..:$PYTHONPATH CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py --model lbvim_small_patch16_224 --batch-size 256 --drop-path 0.05 --weight-decay 0.05 --lr 1e-3 --num_workers 8 --data-path /dev/shm/jzhang/imagenet1k --output_dir  /home/jingwezhang1/result/bivim/vim_IN1k_para/bivim_s_layerflip --no_amp &> /home/jingwezhang1/result/bivim/vim_IN1k_para/bivim_s_layerflip/log.txt

# lbvim 528
PYTHONPATH=.:..:$PYTHONPATH CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py --model lbvim_528_patch16_224_atthead --batch-size 128 --drop-path 0.05 --weight-decay 0.05 --lr 1e-3 --num_workers 8 --data-path /dev/shm/jzhang/imagenet1k --output_dir  /home/jingwezhang1/result/bivim/vim_IN1k_para/bivim_528_layerflip_atthead_qkv --no_amp &> /home/jingwezhang1/result/bivim/vim_IN1k_para/bivim_528_layerflip_atthead_qkv/log.txt