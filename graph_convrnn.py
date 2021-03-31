import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolutionLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907, add trainable mask.
    """

    def __init__(self, in_features, out_features, num_nodes, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # Add mask here, as trainable parameters.
        self.mask = Parameter(torch.ones(num_nodes, num_nodes))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum('bnc,cd -> bnd', input, self.weight)
        output = torch.einsum('mn,bnd -> bmd', adj * self.mask, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, num_nodes, dropout, bias=True, **kwargs):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolutionLayer(nfeat, nhid, num_nodes, bias=bias)
        self.gc2 = GraphConvolutionLayer(nhid, nout, num_nodes, bias=bias)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        x in shape [batch_size, num_node, feat_dim]
        adj in shape [num_node, num_node]
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903, add trainable mask.
    """

    def __init__(self, in_features, out_features, num_nodes, dropout, alpha=0.2, concat=True, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Add mask here, as trainable parameters.
        self.mask = Parameter(torch.ones(num_nodes, num_nodes))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.einsum('bnc,cd -> bnd', h, self.W)  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        # Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        e = self.leakyrelu(torch.einsum('bnmc,cd -> bnmd', a_input, self.a).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.bmm(attention, Wh)
        h_prime = torch.einsum('bnm,bmd -> bnd', attention * self.mask, Wh)   # h.shape: (B, N, out_features)

        if self.concat:
            return F.elu(h_prime) + self.bias if self.bias is not None else F.elu(h_prime)
        else:
            return h_prime + self.bias if self.bias is not None else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N = Wh.size()[:2]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # in shape: (B, N * N, out_features)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        # all_combinations_matrix in shape: (B, N * N, 2 * out_features)

        return all_combinations_matrix.reshape((B, N, N, 2 * self.out_features))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, num_nodes, dropout, nheads, bias=True, **kwargs):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, num_nodes,
                                               dropout=self.dropout, concat=True, bias=bias) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, num_nodes, dropout=dropout, concat=False, bias=bias)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.out_att(x, adj)


class Graph_ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, gnn_hidden_dim, rnn_hidden_dim, num_nodes, gnn_mode, gnn_bias, dropout, gat_nheads,
                 convlstm_formula='simplified'):
        """
        Initialize Graph ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        gnn_hidden_dim: int
            Number of channels of hidden state of GNN.
        rnn_hidden_dim: int
            Number of channels of hidden state of LSTM.
        num_nodes: int
            Number of graph nodes.
        gnn_mode: str
            GNN mode in {'gcn', 'gat'}.
        gnn_bias: bool
            Whether or not to add the bias for GNN.
        dropout: float
            Dropout probability for GNN.
        gat_nheads: int
            Number of GAT heads.
        convlstm_formula:
            'original': use ConvLSTM formula in original paper (https://arxiv.org/pdf/1506.04214.pdf)
            'simplified': use ConvLSTM formula in this repo: https://github.com/ndrplz/ConvLSTM_pytorch
        """

        super(Graph_ConvLSTMCell, self).__init__()
        assert gnn_mode in ['gcn', 'gat']
        assert convlstm_formula in ['original', 'simplified']
        self.input_dim = input_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_nodes = num_nodes
        self.gnn_bias = gnn_bias
        self.dropout = dropout
        self.convlstm_formula = convlstm_formula
        self.gat_nheads = None

        if gnn_mode == 'gcn':
            graph_net = GCN
        else:
            graph_net = GAT
            assert gat_nheads is not None
            self.gat_nheads = gat_nheads

        if self.convlstm_formula == 'simplified':
            self.graph_net = graph_net(nfeat=self.input_dim + self.rnn_hidden_dim,
                                       nhid=self.gnn_hidden_dim,
                                       nout=4 * self.rnn_hidden_dim,
                                       num_nodes=self.num_nodes,
                                       dropout=self.dropout,
                                       nheads=self.gat_nheads,
                                       bias=self.gnn_bias)

        else:  # 'original formula'
            self.graph_nets = [graph_net(nfeat=self.input_dim + t[0] * self.rnn_hidden_dim,
                                         nhid=self.gnn_hidden_dim,
                                         nout=t[1] * self.rnn_hidden_dim,
                                         num_nodes=self.num_nodes,
                                         dropout=self.dropout,
                                         nheads=self.gat_nheads,
                                         bias=self.gnn_bias) for t in [(2, 2), (1, 1), (2, 1)]]
            for i, gnet in enumerate(self.graph_nets):
                self.add_module('graph_net_{}'.format(i), gnet)

    def forward(self, input_tensor, cur_state, adj):
        h_cur, c_cur = cur_state  # B, N, C

        if self.convlstm_formula == 'simplified':
            combined = torch.cat([input_tensor, h_cur], dim=-1)  # concatenate along channel axis
            combined_conv = self.graph_net(combined, adj)  # B, N, Cout

            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.rnn_hidden_dim, dim=-1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)

        else:  # 'original formula'
            combined_1 = torch.cat([input_tensor, h_cur, c_cur], dim=-1)
            combined_conv_1 = self.graph_nets[0](combined_1, adj)
            cc_i, cc_f = torch.split(combined_conv_1, self.rnn_hidden_dim, dim=-1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)

            combined_2 = torch.cat([input_tensor, h_cur], dim=-1)
            cc_g = self.graph_nets[1](combined_2, adj)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g

            combined_3 = torch.cat([input_tensor, h_cur, c_next], dim=-1)
            cc_o = self.graph_nets[2](combined_3, adj)
            o = torch.sigmoid(cc_o)
            h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, num_nodes, device=torch.device("cuda:0")):
        return (torch.zeros(batch_size, num_nodes, self.rnn_hidden_dim, device=device),
                torch.zeros(batch_size, num_nodes, self.rnn_hidden_dim, device=device))


class Graph_ConvGRUCell(nn.Module):

    def __init__(self, input_dim, gnn_hidden_dim, rnn_hidden_dim, num_nodes, gnn_mode, gnn_bias, dropout, gat_nheads):
        """
        Initialize Graph ConvGRU cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        gnn_hidden_dim: int
            Number of channels of hidden state of GNN.
        rnn_hidden_dim: int
            Number of channels of hidden state of GRU.
        num_nodes: int
            Number of graph nodes.
        gnn_mode: str
            GNN mode in {'gcn', 'gat'}.
        gnn_bias: bool
            Whether or not to add the bias for GCN.
        dropout: float
            Dropout probability for GNN.
        gat_nheads: int
            Number of GAT heads.
        """

        super(Graph_ConvGRUCell, self).__init__()
        assert gnn_mode in ['gcn', 'gat']
        self.input_dim = input_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_nodes = num_nodes
        self.gnn_bias = gnn_bias
        self.dropout = dropout
        self.gat_nheads = gat_nheads

        if gnn_mode == 'gcn':
            graph_net = GCN
        else:
            graph_net = GAT
            assert gat_nheads is not None
            self.gat_nheads = gat_nheads

        self.graph_net_1 = graph_net(nfeat=self.input_dim + self.rnn_hidden_dim,
                                     nhid=self.gnn_hidden_dim,
                                     nout=2 * self.rnn_hidden_dim,
                                     num_nodes=self.num_nodes,
                                     dropout=self.dropout,
                                     nheads=self.gat_nheads,
                                     bias=self.gnn_bias)

        self.graph_net_2 = graph_net(nfeat=self.input_dim + self.rnn_hidden_dim,
                                     nhid=self.gnn_hidden_dim,
                                     nout=self.rnn_hidden_dim,
                                     num_nodes=self.num_nodes,
                                     dropout=self.dropout,
                                     nheads=self.gat_nheads,
                                     bias=self.gnn_bias)

    def forward(self, input_tensor, cur_state, adj):
        h_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=-1)  # concatenate along channel axis

        combined_conv = self.graph_net_1(combined, adj)  # B, N, Cout

        gamma, beta = torch.split(combined_conv, self.rnn_hidden_dim, dim=-1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=-1)
        cc_ct = self.graph_net_2(combined, adj)
        ct = torch.tanh(cc_ct)

        h_next = (1 - update_gate) * h_cur + update_gate * ct
        return h_next

    def init_hidden(self, batch_size, num_nodes, device=torch.device("cuda:0")):     # todo: device
        return torch.zeros(batch_size, num_nodes, self.rnn_hidden_dim, device=device)


class Graph_ConvRNN(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        num_layers: Number of RNN layers stacked on each other
        gnn_hidden_dim: Number of hidden channels for GNN
        rnn_hidden_dim: Number of hidden channels for RNN
        num_nodes: Number of graph nodes
        rnn_mode: RNN type in {'lstm', 'gru'}
        gnn_mode: GNN type in {'gcn', 'gat}
        gnn_bias: Bias or no bias in GNN
        gnn_dropout: dropout probability in GNN
        return_all_layers: Return the list of computations for all layers
        batch_first: Whether or not dimension 0 is the batch or not
        gat_nheads: necessary when gnn_mode == 'gat', number of GAT heads

    Input:
        input_tensor: a tensor of size [B, T, N, C] or [T, B, N, C]
        adj: adjacency matrix, a tensor of size [N, N]
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                if rnn_mode == 'lstm', each element of the list is a tuple (h, c) for rnn hidden state and memory
                if rnn_mode == 'gru', each element of the list is h for rnn hidden state
    Example:
        >> device = torch.device("cuda:0")
        >> x = torch.rand((128, 10, 50, 64)).to(device)
        >> adj = torch.rand((50, 50)).to(device)
        >> graph_convrnn = Graph_ConvRNN(64, 2, 32, 16, 50).to(device)
        >> _, last_states = graph_convrnn(x, adj)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, num_layers, gnn_hidden_dim, rnn_hidden_dim, num_nodes,
                 rnn_mode='lstm', gnn_mode='gcn', gnn_bias=True, gnn_dropout=0.5, batch_first=False,
                 return_all_layers=False, **kwargs):
        super(Graph_ConvRNN, self).__init__()
        assert rnn_mode in ['lstm', 'gru'] and gnn_mode in ['gcn', 'gat']

        # Make sure that both `rnn_hidden_dim` and `gcn_hidden_dim` are lists having len == num_layers
        gnn_hidden_dim = self._extend_for_multilayer(gnn_hidden_dim, num_layers)
        rnn_hidden_dim = self._extend_for_multilayer(rnn_hidden_dim, num_layers)
        if not len(gnn_hidden_dim) == len(rnn_hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.gnn_hidden_dim = gnn_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_nodes = num_nodes
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.gnn_dropout = gnn_dropout
        self.rnn_mode = rnn_mode
        self.gnn_mode = gnn_mode
        self.gnn_bias = gnn_bias
        if self.gnn_mode == 'gat':
            assert 'gat_nheads' in kwargs.keys() and kwargs['gat_nheads'] is not None
            self.gat_nheads = kwargs['gat_nheads']

        if self.rnn_mode == 'lstm':
            cell = Graph_ConvLSTMCell
        else:
            cell = Graph_ConvGRUCell

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.rnn_hidden_dim[i - 1]

            cell_list.append(cell(input_dim=cur_input_dim,
                                  rnn_hidden_dim=self.rnn_hidden_dim[i],
                                  gnn_hidden_dim=self.gnn_hidden_dim[i],
                                  num_nodes=self.num_nodes,
                                  dropout=self.gnn_dropout,
                                  gnn_mode=self.gnn_mode,
                                  gnn_bias=self.gnn_bias,
                                  gat_nheads=self.gat_nheads if self.gnn_mode == 'gat' else None))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, adj, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: 4-D Tensor either of shape [B, T, N, C] or [T, B, N, C]
        adj: adjacency matrix， 2-D Tensor of shape (N, N)
        hidden_state: initial hidden state for RNN

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, n, c) -> (b, t, n, c)
            input_tensor = input_tensor.permute(1, 0, 2, 3)

        b, _, n, _ = input_tensor.size()
        if not n == adj.size()[0] == self.num_nodes:
            raise ValueError('Inconsistent number of nodes in graph.')
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, num_nodes=n)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            output_inner = []
            for t in range(seq_len):
                state_list = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                       cur_state=hidden_state[layer_idx],
                                                       adj=adj)
                output_inner.append(state_list[0] if self.rnn_mode == 'lstm' else state_list)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(state_list)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, num_nodes):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, num_nodes))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
