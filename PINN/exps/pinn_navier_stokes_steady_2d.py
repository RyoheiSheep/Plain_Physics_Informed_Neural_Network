import os
from pinn.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.input_size=2
        self.hidden_layers=[20,20,20]
        self.output_size=3
        
        self.Re=10

        self.exp_name=os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        self.lambda_bd=5.0
        self.lambda_res=2.0
