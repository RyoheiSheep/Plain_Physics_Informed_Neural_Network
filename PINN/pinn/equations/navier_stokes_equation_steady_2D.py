import torch

def navier_stokes_equation_steady_2D(outputs, x, y,reynolds):

    lambda_div=1.0
    lambda_momentum_x=2.0
    lambda_momentum_y=2.0

    u = outputs[:, 0]
    v = outputs[:, 1]
    p = outputs[:, 2]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

    v_x = torch.autograd.grad(v, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

    p_x = torch.autograd.grad(p, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    
    div=u_x+u_y
    momentum_x=u*u_x+v*u_y+1/reynolds*p_x+1/reynolds*(u_xx+u_yy)
    momentum_y=u*v_x+v*v_y+1/reynolds*p_y+1/reynolds*(v_xx+v_yy)

    loss_div=torch.mean(div**2)
    loss_momentum_x=torch.mean(momentum_x**2)
    loss_momentum_y=torch.mean(momentum_y**2)
    physics_loss = lambda_div*loss_div +lambda_momentum_x*loss_momentum_x +lambda_momentum_y*loss_momentum_y
    return physics_loss
