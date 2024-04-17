
from datetime import datetime
import argparse
import random
import warnings
from loguru import logger

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from functools import partial

from pinn.exp import Exp,get_exp_by_file
from pinn.net import PINN_2D



def parse_arguments():
    parser=argparse.ArgumentParser(description="Train PINN with specified parameters and config")
    parser.add_argument("--exp_file", type=str,default="experiment.py",help="Path to config python file")
    return parser.parse_args()



@logger.catch
def main(exp:Exp,args):
    
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic=True
    
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=exp.get_model().to(device)
    train_data=exp.get_dataset()
    optimizer=optim.LBFGS(model.parameters(),line_search_fn="strong_wolfe")

    mse_metrics=nn.MSELoss()

    log_dir="logs/"+datetime.now().strftime("%Y%m%d-%H%M%S")
    writer=SummaryWriter(log_dir)



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
    

    def closure(input_bd,output_bd_data,input_res):
        optimizer.zero_grad()
        bd_output=model(input_bd)
        bd_loss=mse_metrics(bd_output[:,:2],output_bd_data)
        res_output=model.compute_physics_loss(input_res)
        res_loss=mse_metrics(res_output,torch.zeros_like(res_output))
        total_loss=exp.lambda_bd*bd_loss+exp.lambda_res*res_loss
        total_loss.backward()
        return total_loss

    for epoch in range(exp.max_epoch):
        loss=optimizer.step(partial(closure,input_bd_train,output_bd_train,input_res_train))
        writer.add_scalar("Loss/train",loss.item(),epoch)

        logger.info(f"Epoch {epoch}:Loss: {loss.item()}")
if __name__=="__main__":
   args=parse_arguments()
   exp=get_exp_by_file(args.experiment_file)
   main(exp,args)

