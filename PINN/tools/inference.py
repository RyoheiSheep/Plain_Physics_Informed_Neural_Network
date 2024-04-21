import argparse
from loguru import logger

import numpy as np
import torch

import matplotlib.pyplot as plt

from pinn.exp import Exp,get_exp_by_file
from pinn.net import PINN_2D

import argparse

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
    domain_data=data["inference_data"]
    x=domain_data["coordinates"][0]
    y=domain_data["coordinates"][1]
    input_data=np.column_stack([x,y])
    input_data=torch.from_numpy(input_data).float().to(device)
    
    with torch.no_grad():
        output=model(input_data)
    output=output.detach().numpy()

    X=x.reshape((exp.nx,exp.ny))
    Y=y.reshape((exp.nx,exp.ny))
    U=output.T[0].reshape((exp.nx,exp.ny))
    V=output.T[1].reshape((exp.nx,exp.ny))

    fig,ax=plt.subplots(figsize=(7,7))
   
    print(X.shape)
    # ax.quiver(X[::10,::10],Y[::10,::10],U[::10,::10],V[::10,::10])
    ax.quiver(X,Y,U,V)
    ax.set_title("Velocity Field")
    plt.show()

if __name__=="__main__":
    args=parse_arguments()
    exp=get_exp_by_file(args.exp_file)
    main(exp,args)
