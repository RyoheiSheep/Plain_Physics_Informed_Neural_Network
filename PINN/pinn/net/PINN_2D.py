import torch
from .NN import FCN


class PINN_2D(FCN):
    def __init__(self, input_size, hidden_size_list, output_size, physics_eq):
        super().__init__(input_size, hidden_size_list, output_size)
        self.physics_eq = physics_eq

    def forward(self, x):
        return super().forward(x)

    def compute_physics_loss(self, inputs):
        x = inputs[:, 0].requires_grad_(True)
        y = inputs[:, 1].requires_grad_(True)
        inputs=torch.vstack([x,y]).T
        pinn_outputs = self.forward(inputs)
        physics_loss = self.physics_eq(pinn_outputs, x, y)
        return physics_loss    
    

if __name__ == "__main__":
    from pinn.equations import laplace_equation
    net = PINN_2D(2, [4, 4, 4], 1, laplace_equation)
    test_input = torch.randn(10, 2)
    test_output = net(test_input)
    test_physics_loss = net.compute_physics_loss(test_input)
    print(f"net: {net}")
    print(f"input: {test_input}")
    print(f"output: {test_output}")
    print(f"physics_loss: {test_physics_loss}")
