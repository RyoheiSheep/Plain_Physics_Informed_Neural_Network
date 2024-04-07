import torch
import numpy as np

def steady_2D_cavity(nx,ny):
    """
    Generate boundary data for a simplified case of cavity flow

    Parameters:
        nx(int) : Number of grid points along x-axis
        ny(int) : Number of grid points along y-axis

    Returns:
        x_boundary(np.ndarray): x-coordinates values for boundary points.
        y_boundary(np.ndarray): y-coordinates values for boundary points.
        u_boundary(np.ndarray): u-velocity values at boundary points.
        v_boundary(np.ndarray): v-velocity values at boundary points.

    """

    x=np.linspace(0,1,nx)
    y=np.linspace(0,1,ny)

    x_boundary=[]
    y_boundary=[]
    u_boundary=[]
    v_boundary=[]

    # Top boundary (moving lid)
    x_boundary.extend(x)
    y_boundary.extend(np.ones_like(x))
    u_boundary.extend(np.ones_like(x))
    v_boundary.extend(np.zeros_like(x))

    # Right boundary (excluding coners)
    x_boundary.extend(np.ones(ny-2))
    y_boundary.extend(y[1:-1])
    u_boundary.extend(np.zeros(ny-2))
    v_boundary.extend(np.zeros(ny-2))

    # Bottom boundary (excluding corners)
    x_boundary.extend(x[::-1][1:-1])
    y_boundary.extend(np.zeros(nx-2))
    u_boundary.extend(np.zeros(nx-2))
    v_boundary.extend(np.zeros(nx-2))

    # Left boundary (excluding corners)
    x_boundary.extend(np.zeros(ny-2))
    y_boundary.extend(y[::-1][1:-1])
    u_boundary.extend(np.zeros(ny-2))
    v_boundary.extend(np.zeros(ny-2))

    return np.array(x_boundary),np.array(y_boundary),np.array(u_boundary),np.array(v_boundary)



if __name__=="__main__":
    x_boundary,y_boundary,u_boundary,v_boundary=steady_2D_cavity(10,10)
    print(x_boundary.dtype)
    print(y_boundary)
    print(u_boundary)

