from dataclasses import dataclasss
from typing import List

@dataclass
class BaseExp:
    input_size:int =3
    hidden_layers:List(int)=[4,4,4]
    output_size:int=1

    physics_eq:str 

    train_epoch:int = 300

    batch_size:int =32

    opt_method:str="LBFGS"

    
