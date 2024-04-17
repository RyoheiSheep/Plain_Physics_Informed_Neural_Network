import argparse
from loguru import logger

import numpy as np
import torch

import matplotlib.pyplot as plt

from pinn.exp import Exp,get_by_file
from pinn.net import PINN_2D

def parse_arguments():
    parser=argparse.ArgumentParser(description="Train PINN with specified parameters and config")
    parser.add_argument("--exp_file", type=str,default="experiment.py",help="Path to config python file")
    parser.add_argument("--ckpt", type=str,default="best_epoch.pth",help="Path to model ckpt to use for inference")
    
    return parser.parse_args()


@logger.catch
def main(exp:Exp,args):
    
    # if exp.seed is not None:
    #     random.seed(exp.seed)
    #     torch.manual_seed(exp.seed)
    #     cudnn.deterministic=True
    
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=exp.get_model().to(device)
    model.eval()
    data=exp.get_dataset()


    data=exp.get_dataset()
    bd_data=data["boundary_data"]
    res_data=data["physics_informed_condition"]
    x_bd=bd_data["coordinates"][0]
    y_bd=bd_data["coordinates"][1]
    u_bd=bd_data["velocity"][0] 
    v_bd=bd_data["velocity"][1] 
    input_bd_train=np.column_stack([x_bd,y_bd])
    input_bd_train=torch.from_numpy(input_bd_train).float().to(device)
    output_bd_train=np.column_stack([u_bd,v_bd])
    output_bd_train=torch.from_numpy(output_bd_train).float().to(device)
    
    x_res=res_data["coordinates"][0]
    y_res=res_data["coordinates"][1]
    input_res_train=np.column_stack([x_res,y_res])
    input_res_train=torch.from_numpy(input_res_train).float().to(device)

    with torch.no_grad():
        output=model(input_res_train)

    