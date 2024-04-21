import numpy as np


def steady_2D_cavity(nx,ny):
    """
    Generate coorrdinates data to calcultate physics-informed loss for a simplified case of cavity flow

    Parameters:
            nx (int): Number of grid points along x-axis.
            ny (int): Number of grid points along y-axis.

        Returns:
            x_boundary (np.ndarray): x-coordinate values for boundary points.
            y_boundary (np.ndarray): y-coordinate values for boundary points.

    """

    x=np.linspace(0,1,nx+2)
    y=np.linspace(0,1,ny+2)
    x=x[1:-1]
    y=y[1:-1]
    xx,yy=np.meshgrid(x,y)


    return {"coordinates":(xx.ravel(),yy.ravel())}


if __name__=="__main__":
    nx=10
    ny=10
    sparcing=0.1

    coordinates=steady_2D_cavity(nx,ny)
    coordinates=coordinates["coordinates"]
    print(coordinates[0].shape)
    # print(f"Sample coordinates data: \n {coordinates}")
