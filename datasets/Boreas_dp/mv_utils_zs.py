"""
this Realistic_Projection function is from https://github.com/yangyangyang127/PointCLIP_V2
and we make some changes based on it
"""

from torch_scatter import scatter
import torch.nn as nn
import numpy as np
import torch
import open3d as o3d

TRANS = -1.5

# realistic projection parameters
PARAMS = {'maxpoolz':1, 
          'maxpoolxy':7, 
          'maxpoolpadz':0, 
          'maxpoolpadxy':2,
          'convz':1, 
          'convxy':3, 
          'convsigmaxy':3, 
          'convsigmaz':1, 
          'convpadz':0, 
          'convpadxy':1,
          'imgbias':0., 
          'depth_bias':0.1, 
          'obj_ratio':0.2, 
          'bg_clr':0.0,
          'resolution': 112, 
          'depth': 8}

OXFORD_PARAMS = {'maxpoolz':1, 
                 'maxpoolxy':7 * 2, 
                 'maxpoolpadz':0 * 2, 
                 'maxpoolpadxy':2 * 2,
                 'convz':1, 
                 'convxy':3 * 2, 
                 'convsigmaxy':3 * 2, 
                 'convsigmaz':1, 
                 'convpadz':0, 
                 'convpadxy':1 * 2,
                 'depth_bias':0.1, 
                 'obj_ratio':2.5, 
                 'bg_clr':100.0,
                 'resolution': 400, 
                 'depth': 100}

OXFORD_PINHOLE_PARAMS_R1 = {'maxpoolx': 4 * 2,
                         'maxpooly': 4 * 2,
                         'maxpoolpadx': 4,
                         'maxpoolpady': 4,
                         'convx':3 * 2,
                         'convy':3 * 2,  
                         'convsigmax':3 * 2,
                         'convsigmay':3 * 2,
                         'convpadx':3,
                         'convpady':3,
                         'ksize': 3 * 2,
                         'ksigma': 3 * 2,
                         'bg_clr': 1.0,
                         }

OXFORD_PINHOLE_PARAMS_R2 = {'maxpoolx': 8 * 2,
                         'maxpooly': 8 * 2,
                         'maxpoolpadx': 8,
                         'maxpoolpady': 8,
                         'convx':7 * 2,
                         'convy':7 * 2,  
                         'convsigmax': 7 * 2,
                         'convsigmay': 7 * 2,
                         'convpadx':7,
                         'convpady':7,
                         'ksize': 7 * 2,
                         'ksigma': 7 * 2,
                         'bg_clr': 1.0,
                         }

OXFORD_PINHOLE_PARAMS_R3 = {'maxpoolx': 12 * 2,
                         'maxpooly': 12 * 2,
                         'maxpoolpadx': 12,
                         'maxpoolpady': 12,
                         'convx':11 * 2,
                         'convy':11 * 2,  
                         'convsigmax': 11 * 2,
                         'convsigmay': 11 * 2,
                         'convpadx': 11,
                         'convpady': 11,
                         'ksize': 11 * 2,
                         'ksigma': 11 * 2,
                         'bg_clr': 1.0,
                         }

OXFORD_PINHOLE_PARAMS_R4 = {'maxpoolx': 16 * 2,
                         'maxpooly': 16 * 2,
                         'maxpoolpadx': 16,
                         'maxpoolpady': 16 ,
                         'convx':15 * 2,
                         'convy':15 * 2,  
                         'convsigmax':15 * 2,
                         'convsigmay':15 * 2,
                         'convpadx':15,
                         'convpady':15,
                         'ksize': 15 * 2,
                         'ksigma': 15 * 2,
                         'bg_clr': 1.0,
                         }


class Grid2Image(nn.Module):
    """A pytorch implementation to turn 3D grid to 2D image. 
       Maxpool: densifying the grid
       Convolution: smoothing via Gaussian
       Maximize: squeezing the depth channel
    """
    def __init__(self, params=PARAMS):
        super().__init__()
        torch.backends.cudnn.benchmark = False

        self.maxpool = nn.MaxPool3d((params['maxpoolz'], params['maxpoolxy'], params['maxpoolxy']), 
                                    stride=1, padding=(params['maxpoolpadz'], params['maxpoolpadxy'], 
                                    params['maxpoolpadxy']))
        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(params['convz'], params['convxy'], params['convxy']),
                                    stride=1, padding=(params['convpadz'],params['convpadxy'],params['convpadxy']),
                                    bias=True)
        
        # gaussion kernel is used to smooth the shape
        kn3d = get3DGaussianKernel(params['convxy'], params['convz'], sigma=params['convsigmaxy'], zsigma=params['convsigmaz'])
        self.conv.weight.data = torch.Tensor(kn3d).repeat(1,1,1,1,1)
        self.conv.bias.data.fill_(0)
        self.bg_clr = params['bg_clr']
            
    def forward(self, x):

        # densify
        x = self.bg_clr - x
        x = self.maxpool(x.unsqueeze(1))

        # Gaussion kernel smooth
        x = self.conv(x)
        img = torch.max(x, dim=2)[0]
        img = img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:,:,None,None]
        img = 1 - img
        img = img.repeat(1,3,1,1)
        return img


class Grid2Image2D(nn.Module):
    """A pytorch implementation to turn 3D grid to 2D image. 
       Maxpool: densifying the grid
       Convolution: smoothing via Gaussian
       Maximize: squeezing the depth channel
    """
    def __init__(self, params=PARAMS):
        super().__init__()
        torch.backends.cudnn.benchmark = False

        self.maxpool = nn.MaxPool2d((params['maxpoolx'], params['maxpooly']), 
                                    stride=1, padding=(params['maxpoolpadx'], 
                                    params['maxpoolpady']))
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(params['convx'], params['convy']),
                                    stride=1, padding=(params['convpadx'],params['convpady']),
                                    bias=True)
        
        # gaussion kernel is used to smooth the shape
        kn2d = get2DGaussianKernel(params['ksize'], params['ksigma'])
        self.conv.weight.data = torch.Tensor(kn2d).repeat(1,1,1,1)
        self.conv.bias.data.fill_(0)
        self.bg_clr = params['bg_clr']
            
    def forward(self, x, device):

        # densify
        x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(device)
        x = self.bg_clr - x
        x = self.maxpool(x)

        # Gaussion kernel smooth
        x = self.conv(x)
        img = x / torch.max(torch.max(x, dim=-1)[0], dim=-1)[0][:,:,None,None]
        img = self.bg_clr - img
        img = np.asarray(img.squeeze(0).squeeze(0).detach().to('cpu'))
        return img


class PCTransformation:
    """For creating point cloud transformations
    """
    def __init__(self):
        _views = np.asarray([
            [[0, - 1 * np.pi / 4, 0], [0, 0, 0]],
            [[0, - 2 * np.pi / 4, 0], [0, 0, 0]],
            [[0, - 3 * np.pi / 4, 0], [0, 0, 0]],
            [[0, - 4 * np.pi / 4, 0], [0, 0, 0]],
            [[0, - 5 * np.pi / 4, 0], [0, 0, 0]],
            [[0, - 6 * np.pi / 4, 0], [0, 0, 0]],
            [[0, - 7 * np.pi / 4, 0], [0, 0, 0]],
            [[0, - 8 * np.pi / 4, 0], [0, 0, 0]]
            ])
        
        #     _views = np.asarray([
        # [[1 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[3 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[5 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[7 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        # ])
        
        # adding some bias to the view angle to reveal more surface
        _views_bias = np.asarray([
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
            ])
        
        #     _views_bias = np.asarray([
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        # ])

        self.num_views = _views.shape[0]
        angle = torch.tensor(_views[:, 0, :]).float()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views_bias[:, 0, :]).float()
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)

        self.translation = torch.tensor(_views[:, 1, :]).float()
        self.translation = self.translation.unsqueeze(1)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat



def points2grid(points, translation, params=PARAMS):
    """Quantize each point cloud to a 3D grid.
    Args:
        points (torch.tensor): of size [B, _, 3]
    Returns:
        grid (torch.tensor): of size [B * self.num_views, depth, resolution, resolution]
    """
    
    batch = points.shape[0]

    # the PointNetVLAD's Oxford Robotcar dataset has already been normalized and centrolized at the ego car's position
    translation = translation.to(points.device)
    points = points - translation
    points[:, :, :2] = points[:, :, :2] * params['obj_ratio']
    
    _x = (points[:, :, 0] + 1) / 2 * params['resolution']
    _y = (points[:, :, 1] + 1) / 2 * params['resolution']
    _z = points[:, :, 2]  * (params['depth'] - 2)

    grid = torch.ones([batch, 
                       params['depth'], 
                       params['resolution'], 
                       params['resolution']], 
                       device=points.device).view(batch, -1) * params['bg_clr']
    for i in range(batch):
        _x_curr = _x[i][_z[i] >= 0]
        _y_curr = _y[i][_z[i] >= 0]
        _z_curr = _z[i][_z[i] >= 0]

        _x_curr = _x_curr.ceil_()
        _y_curr = _y_curr.ceil_()
        z_curr_int = _z_curr.ceil()

        _x_curr = torch.clip(_x_curr, 1, params['resolution'] - 2)
        _y_curr = torch.clip(_y_curr, 1, params['resolution'] - 2)
        _z_curr = torch.clip(_z_curr, 1, params['depth'] - 2)

        coordinates = z_curr_int * params['resolution'] * params['resolution'] + _y_curr * params['resolution'] + _x_curr
        # assign the minimum depth value for each voxel
        grid[i] = scatter(_z_curr, coordinates.long(), dim=-1, out=grid[i], reduce="min")

    grid = grid.reshape((batch, params['depth'], params['resolution'], params['resolution'])).permute((0,1,3,2))

    return grid



class Realistic_Projection:
    """For creating images from PC based on the view information.W
    """
    def __init__(self, device='cpu', params=PARAMS):
        _views = np.asarray([
            [[- 1 * np.pi / 4, 0, 0], [0, 0, 0]],
            [[- 2 * np.pi / 4, 0, 0], [0, 0, 0]],
            [[- 3 * np.pi / 4, 0, 0], [0, 0, 0]],
            [[- 4 * np.pi / 4, 0, 0], [0, 0, 0]],
            [[- 5 * np.pi / 4, 0, 0], [0, 0, 0]],
            [[- 6 * np.pi / 4, 0, 0], [0, 0, 0]],
            [[- 7 * np.pi / 4, 0, 0], [0, 0, 0]],
            [[- 8 * np.pi / 4, 0, 0], [0, 0, 0]]
            ])
        
        #     _views = np.asarray([
        # [[1 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[3 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[5 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[7 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        # [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        # ])
        
        # adding some bias to the view angle to reveal more surface
        _views_bias = np.asarray([
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
            ])
        
        #     _views_bias = np.asarray([
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        # [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
        # ])

        self.num_views = _views.shape[0]
        self.params = params

        angle = torch.tensor(_views[:, 0, :]).float().to(device)
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views_bias[:, 0, :]).float().to(device)
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)

        self.translation = torch.tensor(_views[:, 1, :]).float().to(device)
        self.translation = self.translation.unsqueeze(1)

        self.grid2image = Grid2Image(self.params).to(device)
    
    def get_img(self, points):
        """
        the input points is shape of [B, N, 3]
        it should be previously normalized and centered at the ego car,
        and the depth direction should be the direction of driving
        """
        b, _, _ = points.shape
        v = self.translation.shape[0]

        # transform the point clouds to simulate the view changing
        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1))

        grid = points2grid(points=_points, 
                           translation=self.translation, 
                           params=self.params).squeeze()
        img = self.grid2image(grid)
        return img
    
    @staticmethod
    def point_transform(points, rot_mat, rot_mat2):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param rot_mat2: [batch, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        return points


def get2DGaussianKernel(ksize, sigma=0):
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel


def get3DGaussianKernel(ksize, depth, sigma=2, zsigma=2):
    kernel2d = get2DGaussianKernel(ksize, sigma)
    zs = (np.arange(depth, dtype=np.float32) - depth//2)
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    kernel3d = np.repeat(kernel2d[None,:,:], depth, axis=0) * zkernel[:,None, None]
    kernel3d = kernel3d / torch.sum(kernel3d)
    return kernel3d
        