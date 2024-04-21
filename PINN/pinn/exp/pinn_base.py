import os
import random
from functools import partial
import torch
import torch.nn as nn

from .base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        #model config
        self.input_size=2
        self.hidden_layers=[4,4,4]
        self.output_size=1
        
        self.physics_law="steady_cavity_2D"

        #optimizer config
        self.optimizer="LBFGS"

        # train config
        self.max_epoch=200
        self.eval_interval=10
        self.nx=20
        self.ny=20

        #experiment config
        self.exp_name=os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # data config
        self.data_name="steady_cavity_2D"


    def get_model(self):
        from pinn.net.PINN_2D import PINN_2D

        if self.physics_law=="laplace_equation":
            from pinn.equations import laplace_equation
            self.equation=laplace_equation
        elif self.physics_law=="steady_cavity_2D":
            from pinn.equations import navier_stokes_equation_steady_2D
            
            self.equation=partial(navier_stokes_equation_steady_2D,reynolds=self.Re)

        self.model=PINN_2D(self.input_size,self.hidden_layers,self.output_size,self.equation)
        self.model.train()

        return self.model

    def get_dataset(self):
        from pinn.common import boundary_conditions, physics_informed_conditions,inference_conditions

        if self.data_name=="steady_cavity_2D":
            self.boundary_data=boundary_conditions.steady_2D_cavity(self.nx,self.ny)
            self.physics_informed_data=physics_informed_conditions.steady_2D_cavity(self.nx,self.ny)
            self.inference_data=inference_conditions.steady_2D_cavity(self.nx,self.ny)
        return {"boundary_data":self.boundary_data,"physics_informed_condition":self.physics_informed_data,"inference_data":self.inference_data}

