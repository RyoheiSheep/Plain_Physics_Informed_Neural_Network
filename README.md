# Package of Physics Informed Neural Network

## The contents

This library is for fluid simulation. 
The following physics equations are implemented.

- [x] Laplace equation 
- [] 2D Navier Stokes equation
- [] 1D diffusion equation
- [] 2D diffusion equation


## Installation

Run the following code to install

1. Make a virtual environment and activate it.
```
python -m venv env-PINN
source ./env-PINN/bin/activate

```

2. Install the required packages and the PINN package

```

pip install -r requirements.txt
pip install -e .

```

## How to use

### Construct PINN 

Use the following code to construct a PINN.

```
import torch
from PINN.PINN_2D import PINN
from PINN.PINN2D import laplace_equation

net = PINN(input_size=3,hidden_layers=[4,4,4],output_size=1,physics_eq=laplace_equation)

#prepare input tensor
input=torch.randn(10,2)
#inference 
pinn_output=net(input)
#calculate physics-informed loss
physics_loss=net.compute_physics_loss(input)




```
