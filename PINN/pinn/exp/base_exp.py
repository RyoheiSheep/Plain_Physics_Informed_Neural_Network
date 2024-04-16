from abc import ABCMeta, abstractmethod
from typing import Dict
from torch.nn import Module
import torch

class BaseExp:
    def __init__(self):
        
        self.seed=2024
        self.output_dir="./PINN_outputs"
        self.print_interval=10
        self.eval_interval=5
        self.dataset=None


    @abstractmethod
    def get_model(self)->Module:
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_data_loader(
            self,batch_size:int)-> Dict[str,torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self,batch_size:int)->torch.optim.Optimizer:
        pass

   # @abstractmethod
   # def get_physics_law(self,str):
   #     pass
   #      self.input_size:int =3
   #      self.hidden_layers:List(int)=[4,4,4]
   #      output_size:int=1

   #      physics_eq:str 

   #      train_epoch:int = 300

   #      batch_size:int =32

   #      opt_method:str="LBFGS"

    
