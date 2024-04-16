import torch

def laplace_equation_2D(outputs, x, y):
    u = outputs[:, 0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    physics_loss = torch.mean((u_xx + u_yy) ** 2)
    return physics_loss