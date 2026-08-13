"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn as nn
from torch.autograd import Variable
from .GraphConvInfo import GraphConvInfo
from . import utils

class GraphConvFunction(object):
    """Autograd-native ECC aggregation compatible with modern PyTorch.

    The original implementation used the pre-0.4 stateful ``Function`` API
    and hand-written CUDA kernels. PyTorch 1.9 no longer supports that API.
    This implementation preserves the exact edge filtering and mean
    aggregation while expressing them with differentiable tensor operations.
    """

    def __init__(self, in_channels, out_channels, idxn, idxe, degs, degs_gpu, edge_mem_limit=1e20):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._idxn = idxn
        self._idxe = idxe
        self._degs = degs
        self._degs_gpu = degs_gpu
        self._shards = utils.get_edge_shards(degs, edge_mem_limit)

    def __call__(self, input, weights):
        full_weight_matrix = weights.dim() == 3
        assert full_weight_matrix or (
            self._in_channels == self._out_channels and weights.size(1) == self._in_channels
        )

        outputs = []
        start_node, start_edge = 0, 0
        for num_nodes, num_edges in self._shards:
            source_indices = self._idxn.narrow(0, start_edge, num_edges)
            source_features = torch.index_select(input, 0, source_indices)
            if self._idxe is not None:
                edge_weights = torch.index_select(
                    weights, 0, self._idxe.narrow(0, start_edge, num_edges)
                )
            else:
                edge_weights = weights.narrow(0, start_edge, num_edges)

            if full_weight_matrix:
                products = torch.bmm(source_features.unsqueeze(1), edge_weights).squeeze(1)
            else:
                products = source_features * edge_weights

            degrees = self._degs.narrow(0, start_node, num_nodes).to(input.device)
            target_indices = torch.repeat_interleave(
                torch.arange(num_nodes, device=input.device, dtype=torch.long), degrees
            )
            aggregated = input.new_zeros((num_nodes, self._out_channels))
            aggregated.index_add_(0, target_indices, products)
            outputs.append(aggregated / degrees.clamp(min=1).to(input.dtype).unsqueeze(1))
            start_node += num_nodes
            start_edge += num_edges

        return torch.cat(outputs, dim=0)



class GraphConvModule(nn.Module):
    """ Computes graph convolution using filter weights obtained from a filter generating network (`filter_net`).
        The input should be a 2D tensor of size (# nodes, `in_channels`). Multiple graphs can be concatenated in the same tensor (minibatch).
    
    Parameters:
    in_channels: number of input channels
    out_channels: number of output channels
    filter_net: filter-generating network transforming a 2D tensor (# edges, # edge features) to (# edges, in_channels*out_channels) or (# edges, in_channels)
    gc_info: GraphConvInfo object containing graph(s) structure information, can be also set with `set_info()` method.
    edge_mem_limit: block size (number of evaluated edges in parallel) for convolution evaluation, a low value reduces peak memory. 
    """

    def __init__(self, in_channels, out_channels, filter_net, gc_info=None, edge_mem_limit=1e20):
        super(GraphConvModule, self).__init__()
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._fnet = filter_net
        self._edge_mem_limit = edge_mem_limit
        
        self.set_info(gc_info)
        
    def set_info(self, gc_info):
        self._gci = gc_info
    
    def forward(self, input):       
        # get graph structure information tensors
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()
        edgefeats = Variable(edgefeats, requires_grad=False)
        
        # evalute and reshape filter weights
        weights = self._fnet(edgefeats)
        assert input.dim()==2 and weights.dim()==2 and (weights.size(1) == self._in_channels*self._out_channels or
               (self._in_channels == self._out_channels and weights.size(1) == self._in_channels))
        if weights.size(1) == self._in_channels*self._out_channels:
            weights = weights.view(-1, self._in_channels, self._out_channels)

        return GraphConvFunction(self._in_channels, self._out_channels, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(input, weights)
        





class GraphConvModulePureAutograd(nn.Module):
    """
    Autograd-only equivalent of `GraphConvModule` + `GraphConvFunction`. Unfortunately, autograd needs to store intermediate products, which makes the module work only for very small graphs. The module is kept for didactic purposes only.
    """

    def __init__(self, in_channels, out_channels, filter_net, gc_info=None):
        super(GraphConvModulePureAutograd, self).__init__()
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._fnet = filter_net
        
        self.set_info(gc_info)
        
    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, input):
        # get graph structure information tensors
        idxn, idxe, degs, edgefeats = self._gci.get_buffers()
        idxn = Variable(idxn, requires_grad=False)
        edgefeats = Variable(edgefeats, requires_grad=False)
        
        # evalute and reshape filter weights
        weights = self._fnet(edgefeats)
        assert input.dim()==2 and weights.dim()==2 and weights.size(1) == self._in_channels*self._out_channels
        weights = weights.view(-1, self._in_channels, self._out_channels)
            
        # select sequence of matching pairs of node and edge weights            
        if idxe is not None:
            idxe = Variable(idxe, requires_grad=False)
            weights = torch.index_select(weights, 0, idxe)        
        
        sel_input = torch.index_select(input, 0, idxn)

        # compute matrix-vector products
        products = torch.bmm(sel_input.view(-1,1,self._in_channels), weights)
        
        output = Variable(input.data.new(len(degs), self._out_channels))
        
        # average over nodes
        k = 0
        for i in range(len(degs)):
            if degs[i]>0:
                output.index_copy_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), torch.mean(products.narrow(0,k,degs[i]), 0).view(1,-1))
            else:
                output.index_fill_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), 0)
            k = k + degs[i]

        return output
    
