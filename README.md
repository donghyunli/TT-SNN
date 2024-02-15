# TT-SNN: Tensor Train Decomposition for Efficient Spiking Neural Network Training [[Paper]](https://arxiv.org/pdf/2401.08001.pdf)

Pytorch code for "TT-SNN: Tensor Train Decomposition for Efficient Spiking Neural Network Training" - DATE2024 (Link)

## Dependancy
- Python 3.9
- Pytorch 1.10.0
- Spikingjelly 0.0.0.0.12
- tensorly

## Training Details
(1) **TT_decomposition.py**: Get TT-ranks and save the TT-ranks at checpoint directory

    python TT_decomposition.py --depth 18 --save_dir /checkpoint

(2) **main.py**: Train TT-SNN with saved TT-ranks

    python main.py --depth 18 --dataset CIFAR10 --T 4 --tt_mode PTT --rank_path /rank_checkpoint

- T: time-step (default 4)
- tt_mode: TT modules like STT (Sequential TT), PTT (Parallel TT), and HTT (Half TT)
- rank_path: The checkpoint path of TT-ranks after step (1)

## Others
Data and checkpoint directory can be set in "conf/global_settings.py"


## Reference
Our code is based on below repositories
- VBMF method: https://github.com/CasvandenBogaard/VBMF
- TT decomposition: https://github.com/jacobgil/pytorch-tensor-decompositions


