import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import faiss
from torch.utils.data import DataLoader, SubsetRandomSampler


class NetVLAD(nn.Module):
    # the code source is https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/models/PointNetVlad.py
    # this is used specially for cloud
    def __init__(self, cfgs, out_dim):
        super(NetVLAD, self).__init__()
        self.feature_size = out_dim
        self.output_dim = out_dim
        self.gating = cfgs.gating
        self.add_batch_norm = cfgs.add_batch_norm
        self.cluster_size = cfgs.cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            self.feature_size, self.cluster_size) * 1 / math.sqrt(self.feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            self.feature_size, self.cluster_size) * 1 / math.sqrt(self.feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(self.cluster_size * self.feature_size, self.output_dim) * 1 / math.sqrt(self.feature_size))

        if self.add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(self.cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                self.cluster_size) * 1 / math.sqrt(self.feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(self.output_dim)

        if self.gating:
            self.context_gating = GatingContext(
                self.output_dim, add_batch_norm=self.add_batch_norm)


    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.cluster_weights2 = nn.Parameter(torch.from_numpy(centroids.transpose()))
        self.cluster_weights = nn.Parameter(torch.from_numpy(alpha*centroids_assign.transpose()))
        # self.conv.bias = None


    def initialize_netvlad_layer(self, num_workers, device, cluster_ds, backbone, features_dim):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, num_workers=num_workers,
                                batch_size=1, sampler=random_sampler)
        with torch.no_grad():
            backbone = backbone.eval()
            print("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, features_dim), dtype=np.float32)
            for iteration, inputs in enumerate(random_dl):
                inputs = inputs['image'].to(device)
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * 1 * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(features_dim, self.cluster_size, niter=100, verbose=False)
        kmeans.train(descriptors)
        print(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(device)

    def forward(self, x):
        """
        Args:
            x: (B, feature_size, N) for pc , (B, feature_size, H, W) for image
        """
        B, C = x.shape[:2]
        assert C == self.feature_size
        x = x.reshape(B, self.feature_size, -1) # (B, feature_size, N) or (B, feature_size, H*W)
        _, _, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, feature_size) or (B, H*W, feature_size)
        activation = torch.matmul(x, self.cluster_weights) # cluster_weights is the weight of cluster weight option (B, N, cluster_size)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         N, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, N, self.cluster_size)) # (B, N, cluster_size)

        a_sum = activation.sum(-2, keepdim=True) # (B, 1, cluster_size)
        a = a_sum * self.cluster_weights2 # cluster_weights_2 is the centrol vector of each cluster,it will be updated via the learning process  (B, feature_size, cluster_size)

        activation = torch.transpose(activation, 2, 1) # (B, cluster_size, N)
        vlad = torch.matmul(activation, x) # (B, cluster_size, feature_size)
        vlad = torch.transpose(vlad, 2, 1).contiguous() # (B, feature_size, cluster_size)
        vlad = vlad - a # (B, feature_size, cluster_size)

        # above outpue the Vk(P')

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights) # (B, output_dim)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class NetVLAD_mask(nn.Module):
    # the code source is https://github.com/cattaneod/PointNetVlad-Pytorch/blob/master/models/PointNetVlad.py
    # this is used specially for cloud
    def __init__(self, cfgs, out_dim):
        super(NetVLAD_mask, self).__init__()
        self.feature_size = out_dim
        self.output_dim = out_dim
        self.gating = cfgs.gating
        self.add_batch_norm = cfgs.add_batch_norm
        self.cluster_size = cfgs.cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            self.feature_size, self.cluster_size) * 1 / math.sqrt(self.feature_size))
        if self.add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(self.cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                self.cluster_size) * 1 / math.sqrt(self.feature_size))
            self.bn1 = None
        self.cluster_weights2 = nn.Parameter(torch.randn(
            self.feature_size, self.cluster_size) * 1 / math.sqrt(self.feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(self.cluster_size * self.feature_size, self.output_dim) * 1 / math.sqrt(self.feature_size))

        self.bn2 = nn.BatchNorm1d(self.output_dim)

        if self.gating:
            self.context_gating = GatingContext(
                self.output_dim, add_batch_norm=self.add_batch_norm)


    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.cluster_weights2.data.copy_(torch.from_numpy(centroids.transpose()))
        self.cluster_weights.data.copy_(torch.from_numpy(alpha*centroids_assign.transpose()))
        # self.conv.bias = None


    def forward(self, x, mask):
        """
        Args:
            x: (B, feature_size, N) for pc , (B, feature_size, H, W) for image
            mask: (B, N) for pc, (B, H, W) for image
        """
        B, C = x.shape[:2]
        assert C == self.feature_size
        device = x.device
        mask = mask.reshape(B, -1) # (B, N) or (B, H*W)
        x = x.reshape(B, self.feature_size, -1) # (B, feature_size, N) or (B, feature_size, H*W)
        _, _, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, feature_size) or (B, H*W, feature_size)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        activation_0 = torch.matmul(x, self.cluster_weights) # cluster_weights is the weight of cluster weight option (B, N, cluster_size)

        if self.add_batch_norm:
            activation_0 = activation_0.reshape(-1, self.cluster_size) # (B * N, cluster_size)
            activation_1 = torch.flatten(activation_0) # (B * N * cluster_size)
            with torch.no_grad():
                mask1 = ((torch.flatten(mask)).type(dtype=torch.LongTensor)).to(device) # (B * N)
            real_indices = torch.nonzero(mask1, as_tuple=True)[0] # (num, )
            real_activation = torch.index_select(activation_0, 0, mask1) # (num, cluster_size)
            real_activation = self.bn1(real_activation) # (num, cluster_size)
            real_activation = torch.flatten(real_activation) # (num * cluster_size)
            with torch.no_grad():
                real_indices = torch.repeat_interleave(real_indices * self.cluster_size, self.cluster_size) + (torch.arange(self.cluster_size, device=device)).repeat(real_indices.shape[0]) # (num * cluster_size)
            activation_1.index_put_([real_indices], real_activation) # (B * N * cluster_size)
            activation = activation_1.view(-1,
                                         N, self.cluster_size) # (B, N, cluster_size)
        else:
            activation = activation_0 + self.cluster_biases  # (B, N, cluster_size)
        activation = self.softmax(activation) # (B, N, cluster_size)
        activation = activation.masked_fill(~mask.unsqueeze(-1), 0.0)

        a_sum = activation.sum(-2, keepdim=True) # (B, 1, cluster_size)
        a = a_sum * self.cluster_weights2 # cluster_weights_2 is the centrol vector of each cluster,it will be updated via the learning process  (B, feature_size, cluster_size)

        activation = torch.transpose(activation, 2, 1) # (B, cluster_size, N)
        vlad = torch.matmul(activation, x) # (B, cluster_size, feature_size)
        vlad = torch.transpose(vlad, 2, 1).contiguous() # (B, feature_size, cluster_size)
        vlad = vlad - a # (B, feature_size, cluster_size)

        # above outpue the Vk(P')

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights) # (B, output_dim)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation