import numpy as np


def pc_transform(pc, T):
    """Transforms points given a transform, T. x_out = np.matmul(T, pc)
    Args:
        pc (np.ndarray): Nx3 points
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        points (np.ndarray): The transformed points
    """
    assert T.shape[0] == 4 and T.shape[1] == 4
    p = np.hstack((pc[:, :3], np.ones((pc.shape[0], 1))))
    pc[:, :3] = np.matmul(p, T.transpose())[:, :3]
    return pc

def project_onto_image(self, P, pc, width=2448, height=2048, checkdims=False):
    x = np.hstack((pc[:, :3], np.ones((pc.shape[0], 1))))
    x /= x[:, 2:3]
    x[:, 3] = 1
    x = np.matmul(x, P.transpose())
    if checkdims:
        mask = np.where(
            (x[:, 0] >= 0)
            & (x[:, 0] <= width - 1)
            & (x[:, 1] >= 0)
            & (x[:, 1] <= height - 1)
        )
    else:
        mask = np.ones(x.shape[0], dtype=bool)
    x = x[mask]
    return x