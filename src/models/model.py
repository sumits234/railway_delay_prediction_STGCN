############################
# imports
############################
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
############################
# STGCN Building Blocks
############################


class TemporalConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kt=3):
        """
        Args:
            c_in (int): number of input channels per node
            c_out (int): number of output channels per node
            kt (int, optional): size of temporal kernel (1D convolution). Defaults to 3.
        """
        super(TemporalConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        # for input size NCWH, kernel acts on the WH channels
        self.conv1 = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kt, 1), padding = (1, 0))
        self.conv_reshape = nn.Conv2d(c_in, c_out, kernel_size=(kt, 1), padding = (1, 0))

    def forward(self, X):
        """
        Args:
            X (Torch.Tensor): X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
        Returns:
            [Torch.Tensor]: out shape  = (batch_size, c_out, n_timesteps_in, n_nodes)
        """
        x_residual = X
        # change size of channel dimension of input so it works as part of GLU

        if self.c_in < self.c_out:
            n_pad_total = self.c_out - x_residual.shape[1]
            n_pad_start = n_pad_total // 2
            n_pad_end = n_pad_total - n_pad_start
            
            # Pad on the channel dimension (dim=1)
            # We permute to (B, T, N, C) to pad C, then permute back
            x_residual = x_residual.permute(0, 2, 3, 1).contiguous()
            pad = nn.ZeroPad2d((n_pad_start, n_pad_end, 0, 0)) # Pad left/right (which is now C dim)
            x_residual = pad(x_residual)
            x_residual = x_residual.permute(0, 3, 1, 2).contiguous()
            # x_residual shape = (batch_size, c_out, n_timesteps_in, n_nodes)

        elif self.c_in > self.c_out:
            x_residual = self.conv_reshape(x_residual)

        x_conv = self.conv1(X)
        # x_conv shape = (batch_size, 2*c_out, n_timesteps_in, n_nodes)

        # cut along the "channels" dimension to form P, Q for GLU activation
        P = x_conv[:, 0:self.c_out, :, :]
        Q = x_conv[:, -self.c_out:, :, :]

        out = (P + x_residual) * torch.sigmoid(Q)
        return out


class SpatialConvLayer(nn.Module):
    def __init__(self, c_in, c_out, device, ks=5):
        """
        Args:
            c_in (int): number of input channels per node
            c_out ([type]): number of output channels per node
            ks (int, optional): size of spatial kernel. Defaults to 5.
        """

        super(SpatialConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = ks
        self.device = device

        self.theta = nn.Parameter(torch.FloatTensor(ks * c_in, c_out)).double().to(device)
        self.conv_reshape = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.init_parameters()

    def init_parameters(self):
        # kaiming uniform initialization
        std = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-std, std)


    def graph_conv(self, X, theta, graph_kernel, ks, c_in, c_out):
        """performs graph convolution operation
        """
        batch_size, n_timesteps_in = X.shape[0], X.shape[2]
        n_nodes = graph_kernel.shape[0]
        
        # X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
        X = X.permute(0, 2, 3, 1).contiguous() 
        # X shape = (batch_size, n_timesteps_in, n_nodes, c_in)
        X = X.reshape(-1, c_in)
        # X shape = (batch_size * n_timesteps_in * n_nodes, c_in)
        
        # This part is different from original, seems more efficient
        # n_nodes = graph_kernel.shape[0]
        x_conv = X.reshape(-1, n_nodes)
        # x_conv shape  = [batch_size * n_timesteps_in * c_in, n_nodes]

        # spatial convolution
        x_conv = torch.mm(x_conv, graph_kernel)
        # x_conv  shape  = [batch_size * n_timesteps_in * c_in, ks * n_nodes]

        # reshape 2
        x_conv = x_conv.reshape(-1, c_in * ks)
        # x_conv shape  = [batch_size * n_timesteps_in * n_nodes, c_in * ks]

        # multiply by learned params
        x_conv = torch.mm(x_conv, theta)
        # x_conv shape  = [batch_size * n_timesteps_in * n_nodes, c_out]

        # reshape 3
        x_conv = x_conv.reshape(-1, n_timesteps_in, n_nodes, self.c_out)
        # x_conv shape  = (batch_size, n_timesteps_in, n_nodes, c_out)
        
        x_conv = x_conv.permute(0, 3, 1, 2).contiguous()
        # x_conv shape = (batch_size, c_out, n_timesteps_in, n_nodes)

        return x_conv


    def forward(self, X, graph_kernel):
        """
        Args:
            X (Torch.Tensor): input data, X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
            graph_kernel (Torch.Tensor): Chebyshev or 1st order approximation of scaled graph Laplacien, graph_kernel shape  = (n_nodes, ks * n_nodes)
        Returns:
            [Torch.Tensor]: out shape  = (batch_size, n_timesteps_in, n_nodes, c_out)
        """
        x_residual = X
        
        # change size of channel dimension of input so it works as part of GLU
        if self.c_in < self.c_out:
            n_pad_total = self.c_out - x_residual.shape[1]
            n_pad_start = n_pad_total // 2
            n_pad_end = n_pad_total - n_pad_start
            
            x_residual = x_residual.permute(0, 2, 3, 1).contiguous()
            pad = nn.ZeroPad2d((n_pad_start, n_pad_end, 0, 0)) 
            x_residual = pad(x_residual)
            x_residual = x_residual.permute(0, 3, 1, 2).contiguous()

        elif self.c_in > self.c_out:
            x_residual = self.conv_reshape(x_residual)

        x_conv = self.graph_conv(X, self.theta, graph_kernel, self.ks, self.c_in, self.c_out)
        # x_conv shape  = (batch_size, c_out, n_timesteps_in, n_nodes)

        # add outputs, relu
        out = F.relu(x_conv + x_residual)

        return out


class OutputLayer(nn.Module):
    def __init__(self, channels, n_timesteps_in, n_timesteps_out, kt=3):
        """
        Args:
            channels (array): channel size for output, channels[0] must be equal to the channel size of input to this layer, channels[1] is number of output features for network
            n_timesteps_in (int): number of timesteps in the input data
            n_timesteps_out (int): number of timesteps in the labeled data
            kt (int, optional): size of temporal kernel. Defaults to 3.
        """
        super(OutputLayer, self).__init__()
        c_in, c_out = channels
        self.temporal_conv_layer = TemporalConvLayer(n_timesteps_in, n_timesteps_out, kt=kt).double()
        self.fc = nn.Conv2d(c_in, c_out, kernel_size = (1, 1)).double()

    def forward(self, X):
        """
        Args:
            X (Torch.Tensor): X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
        Returns:
            Torch.Tensor: out shape  = (batch_size, c_out, n_timesteps_out, n_nodes)
        """
        # reduce the time dimension to be same as n_timesteps_out
        X = X.permute(0, 2, 1, 3).contiguous()
        # X shape = (batch_size, n_timesteps_in, c_in, n_nodes)
        out = self.temporal_conv_layer(X)
        # out shape = (batch_size, n_timesteps_out, c_in, n_nodes)
        out = out.permute(0, 2, 1, 3).contiguous()
        # out shape  = (batch_size, c_in, n_timesteps_out, n_nodes)

        # reduce to 1 output feature per graph node
        out = self.fc(out)
        # out shape  = (batch_size, n_features_out, n_timesteps_out, n_nodes)
        return out


############################
# ST-Conv Block
############################


class STConvBlock(nn.Module):
    def __init__(self, channels, n_nodes, device, ks=5, kt=3, drop_prob=0.0):
        super(STConvBlock, self).__init__()
        c_in, c_hid, c_out = channels
        self.temporal_layer_1 = TemporalConvLayer(c_in, c_hid, kt=kt).double()
        self.spatial_layer = SpatialConvLayer(c_hid, c_hid, device, ks=ks).double()
        self.temporal_layer_2 = TemporalConvLayer(c_hid, c_out, kt=kt).double()

        # layer norm is batch norm along a different dimension
        self.layer_norm = nn.LayerNorm([n_nodes, c_out]).double()
        self.dropout = nn.Dropout2d(drop_prob)


    def forward(self, X, graph_kernel):
        """
        Args:
            X (Torch.Tensor): input data, X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
            graph_kernel (Torch.Tensor): Chebyshev or 1st order approximation of scaled graph Laplacien, graph_kernel shape  = (n_nodes, ks * n_nodes)

        Returns:
            Torch.Tensor: out shape  = (batch_size, c_out, n_timesteps_in, n_nodes)
        """
        out = self.temporal_layer_1(X)
        out = self.spatial_layer(out, graph_kernel)
        out = self.temporal_layer_2(out)
        
        # Layer Norm
        # out shape = (batch, c_out, n_timesteps_in, n_nodes)
        out = out.permute(0, 2, 3, 1).contiguous()
        # out shape = (batch, n_timesteps_in, n_nodes, c_out)
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        # out shape = (batch, c_out, n_timesteps_in, n_nodes)
        
        out = self.dropout(out)

        return out


############################
# STGCN Model
############################


class STGCN(nn.Module):
    def __init__(self, blocks, n_timesteps_in, n_timesteps_out, n_nodes, device, ks=5, kt=3, drop_prob=0.0):
        """STGCN Model as implemented in https://arxiv.org/abs/1709.04875
        """
        super(STGCN, self).__init__()

        self.st_conv_block_1 = STConvBlock(blocks[0], n_nodes, device, ks, kt, drop_prob=drop_prob)
        self.st_conv_block_2 = STConvBlock(blocks[1], n_nodes, device, ks, kt, drop_prob=drop_prob)
        # use the "individual" inf mode from the original model
        self.output_layer = OutputLayer(blocks[2], n_timesteps_in, 1)


    def forward(self, X, graph_kernel):
        """
        Args:
            X (Torch.Tensor): input data, X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
            graph_kernel (Torch.Tensor): Chebyshev or 1st order approximation of scaled graph Laplacien, graph_kernel shape  = (n_nodes, ks * n_nodes)
        Returns:
        Torch.Tensor: model prediction, shape  = (batch_size, n_features_out, 1, n_nodes)
        use the "individual" inf mode from the original model
        """
        out = self.st_conv_block_1(X, graph_kernel)
        out = self.st_conv_block_2(out, graph_kernel)
        y_hat = self.output_layer(out)

        return y_hat