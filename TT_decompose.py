import argparse
import torch
import torch.nn as nn

from models.MS_ResNet import resnet18, resnet34
from decompose.decompse_layer import decompose_layer1, decompose_layer2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, required=True, help='network depth')
    parser.add_argument('--save_dir', type=str, required=True, help='save directory for TT-ranks')
    parser.add_argument('--decompose_mode', type=str, default='tt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_class = 100

    # load original model
    if args.depth == 18:
        model = resnet18(num_class)
    elif args.depth == 34:
        model = resnet34(num_class)

    model = model.to(device)
    model.eval()
    model.cpu()
    rank_list = []
    for n, m in model.named_children():        
        for a, b in m.named_children():
            num_children = sum(1 for i in b.children())
            if num_children != 0 and n != 'conv1':
                # in a layer of resnet
                layer = getattr(m, a)
                bottleneck = layer.residual_function 

                conv1 = getattr(bottleneck, "1")
                conv2 = getattr(bottleneck, "4")

                # decompose current conv2d layer with CP/Tucker
                new_layer1, rank1 = decompose_layer1(args.decompose_mode, conv1)
                new_layer2, rank2 = decompose_layer2(args.decompose_mode, conv2)
                rank_list.append(rank1)
                rank_list.append(rank2)

                # set old layer to new and delete
                setattr(bottleneck, "1", nn.Sequential(*new_layer1))
                setattr(bottleneck, "4", nn.Sequential(*new_layer2))
                del conv1
                del conv2
                del bottleneck
                del layer
        torch.save(
            {   'model': model,
                'model_state_dict': model.state_dict(), 
                'rankList': rank_list
            }, args.save_dir
        )
    print(f"finished {args.decompose_mode} decomposition")
