import torch
import numpy as np

def steady_2D_cavity(nx,ny):
    """
    Generate domain coordinates for a simplified case of cavity flow

    Parameter:
        nx(int):Number of grid points aling x-axis
        ny(int):Number of grid points aling y-axis

    Returns:
        x_domain(np.ndarray): x-coordinates for domain points.
        y_domain(np.ndarray): y-coordinates for domain points.
    """

    x=np.linspace(0,1,nx)
    y=np.linspace(0,1,ny)

    xx,yy=np.meshgrid(x,y)


    return {"coordinates": (xx.ravel(),yy.ravel())}

if __name__=="__main__":
    nx=10
    ny=10
    domain_coordinates=steady_2D_cavity(nx,ny)
    data=domain_coordinates["coordinates"][0]
    print(f"coordinates grid shape {data.shape}")


