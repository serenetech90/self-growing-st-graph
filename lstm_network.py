'''
 Class lstm_network to define and intialize lstm based model
'''
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn import functional as F
import numpy as np
import criterion as cr
import grow_graph as sg
''' 
the spatio-temporal lstm network has simplified design choice, by assigning LSTMs only at the node.
excluding edge lstms from this deign means that the graph is node-centric. 
TODO: we will check the choice of interaction-centric design.
'''

batch_size = 4

class HumanNode(nn.Module):
    def __init__(self,args, file_hook=None):
        super(HumanNode, self).__init__()
        # self.lstm_size = args.args.lstm_size
        self.human_rnn_size = args.human_node_rnn_size
        self.human_node_input_size = args.human_node_input_size
        self.human_node_output_size = args.human_node_output_size
        self.human_node_embedding_size = args.human_node_embedding_size
        self.max_len = args.max_len

        # Change lower/upper bounds to see how it fits wider range of dynamics change rates
        self.lower_bound = -1/ self.human_node_embedding_size #2
        self.upper_bound = 1/ self.human_node_embedding_size

        self.relu = nn.ReLU() #nn.PReLU(init=0.4) #nn.ReLU()
        self.softmax = nn.Softmax()
        self.softLog = nn.LogSoftmax(dim=1)
        self.num_layers = args.num_layers

        # if not np.any([self.win, self.w_edge, self.who, self.who2, self.who3]):
        self.vel_gru = nn.GRU(input_size=1, hidden_size=self.human_rnn_size,
                              num_layers=self.num_layers, batch_first=False)

        # if not np.any([self.win, self.w_edge, self.who, self.who2, self.who3]):
        self.register_parameter('win', None)
        self.register_parameter('who', None)
        self.register_parameter('w_edge', None)

        self.file_hook = file_hook

    def reset_parameters(self, input):
        mean = 0
        std = 0.5
        bound = self.lower_bound
        self.win = nn.Parameter(torch.Tensor(self.human_node_embedding_size, 2)
                                .normal_(mean, std).requires_grad_())
        self.in_bias = nn.Parameter(
            torch.Tensor(1, self.human_node_embedding_size).uniform_(-bound, bound).requires_grad_())

        self.w_edge = nn.Parameter(torch.Tensor(self.human_node_embedding_size, input.size()[0])
                                   .normal_(mean,std)
                                   .requires_grad_())
        self.w_edge_tmp = nn.Parameter(torch.Tensor(self.human_node_embedding_size, input.size()[0]) #
                                   .normal_(mean,std)
                                   .requires_grad_())

        self.w_edge_dist = nn.Parameter(torch.Tensor(self.human_node_embedding_size, input.size()[1])
                                   .normal_(mean,std)
                                   .requires_grad_())

        self.ih_bias = nn.Parameter(torch.Tensor(1, self.human_node_embedding_size)
                                    .uniform_(-bound, bound).requires_grad_())

        # self.who = nn.Parameter(torch.Tensor(1, self.human_rnn_size).uniform_(self.lower_bound,self.upper_bound).requires_grad_())
        self.who = nn.Parameter(torch.Tensor(2, self.human_rnn_size).normal_(mean,std)
                                .requires_grad_())

        self.out_bias = nn.Parameter(torch.Tensor(1, 2).uniform_(-bound, bound))

        self.register_parameters()

    def register_parameters(self):
        self.register_parameter(name='win', param=self.win)
        self.register_parameter(name='in_bias', param=self.in_bias)

        self.register_parameter(name='who', param=self.who)
        self.register_parameter(name='out_bias', param=self.out_bias)

        self.register_parameter(name='w_edge', param=self.w_edge)
        self.register_parameter(name='ih_bias', param=self.ih_bias)

    def reset_dynamic_parameter(self, input):
        self.w_edge_c = nn.Parameter(self.w_edge[:, 0:len(input)])
        self.register_parameter('w_edge_weight', self.w_edge_c)

    def forward(self, position, distance, velocity ,h_prev , c_prev, differ=False, updateAdj=False,adj_mat=None,dist_mat=None, retain=True):
        # encoded_input = self.encoding_layer(position)
        self.layer_width = position.size()[0]
        c_curr = torch.zeros(1, 256)

        # .uniform_(self.lower_bound,self.upper_bound)
        x = [self.win, self.w_edge, self.who]  # , self.who2, self.who3
        y = None
        if np.all([a is None for a in x]) or not retain:
            self.reset_parameters(input=distance)
            self.w_edge_c = self.w_edge

        if differ:
            self.reset_dynamic_parameter(input=distance)

        # used the functional linear transformation from nn.functional
        # as we are expecting a variable input length
        encoded_input = F.linear(input=position, weight=self.win, bias=self.in_bias)

        encoded_input = self.relu(encoded_input) #self.relu(encoded_input) BUG: PRelu turns negative values into NaN

        embedded_edge_temp = F.linear(input=distance.t(),weight= self.w_edge_tmp, bias=self.ih_bias)# randn needs to be bounded by smaller interval

        embedded_edge = self.relu(embedded_edge_temp)# relu with randn generate higher sparsity matrix, use prelu better

        if encoded_input.size()[1] > self.max_len:
            self.max_len = encoded_input.size()[1]

        if updateAdj:
            # dist_mat = dist_mat.reshape(shape=(dist_mat.size()[0]*dist_mat.size()[1], 1))
            # self.reset_dynamic_parameter(input=dist_mat)

            distance = torch.sum(distance, dim=1)
            # dist_mat = dist_mat + distance
            # edges_vector = sg.evaluate_edges(embedded_rep=dist_mat)
            embedded_edge_all = F.linear(input=dist_mat, weight=self.w_edge_dist)
            # embedded_edge_all = torch.cat((embedded_edge_spat,embedded_edge_temp),dim=0)
            # embedded_edge_spat + embedded_edge_temp

            adj_mat = sg.update_adjecancy_matrix(adj_mat= adj_mat, dist_rep=dist_mat, #distance.squeeze()
                                                 edge_vec=embedded_edge_all)

            selected_embeddings = torch.zeros((self.max_len, self.human_node_embedding_size))
            # other = torch.zeros((20,20))

            z = adj_mat > 0
            i = 0

            for k in range(len(adj_mat)):
                if k < len(position):
                    sel = embedded_edge_all[z[k]]
                    r1 = len(range(i, k+len(sel)))
                    # if r1 == 0 or r1 < len(embedded_edge_all[z[k]]):
                    end = len(sel) - r1
                    if len(selected_embeddings[i: i + r1]) == len(sel):
                        selected_embeddings[i: i + r1] = sel
                    elif len(selected_embeddings[i: i + r1]) < len(sel):
                        selected_embeddings[i: i + r1 + end] = sel[0:len(selected_embeddings[i: i + r1 + end])]
                    else:
                        selected_embeddings[i: i + r1 + 1] = sel
                    i = k + len(sel) + 1

            embedded_edge = selected_embeddings.unsqueeze(0)

        # embedded_edge_all = embedded_edge_all.masked_select((adj_mat[:] > 0))
        # embedded_edge = selected_embeddings
        # concat_input = torch.cat((encoded_input, selected_embeddings.unsqueeze(0)), 0)
        # input_size = concat_input.size()[1]
        # concat_input = concat_input.unsqueeze(dim=0)
        # TODO: layer for embedding spatial edges (possibly pooling layer ?) by 15th march
        # TODO: learnable criterion for growing spatial edges(growing spatial hidden feature vectors in width and depth)
        # by 10th March xxx by 15th march
        # concat_input = torch.cat((encoded_input, embedded_edge), 0)
        # concat_input = torch.cat((encoded_input,embedded_edge),1) + torch.zeros((h_prev.size()))
        else:
            embedded_edge = embedded_edge.unsqueeze(0)

        encoded_input = torch.cat((encoded_input , torch.zeros((len(encoded_input), abs(encoded_input.size()[1] - self.max_len),
                                                               encoded_input.size()[2]))), dim=1)

        h_prev = torch.cat((h_prev, torch.zeros((len(h_prev), abs(h_prev.size()[1] - self.max_len),h_prev.size()[2]))), dim=1)
        velocity = torch.cat((velocity , torch.zeros(self.max_len - len(velocity), velocity.size()[1])), dim=0)

        concat_input = torch.cat((encoded_input, embedded_edge), 0)
        input_size = concat_input.size()[1]

        # if concat_input.size()[1] > self.max_len:
        #     concat_input = torch.cat((concat_input, torch.zeros(
        #         (len(concat_input), (concat_input.size()[1] - self.max_len), self.human_rnn_size))), dim=1)
        # self.max_len = concat_input.size()[1]
        # concat_input = concat_input.view(concat_input.size()[0], concat_input.size()[1], 1)
        # at_input = torch.zeros(concat_input.size()[0], concat_input.size()[1], self.layer_width + 1) + concat_input
        # self.cell = nn.LSTM(self.layer_width + 1, self.human_rnn_size, num_layers=self.layer_width + 1)
        # self.cell(at_input, (h_prev, c_prev))
        # concat_input = torch.zeros(concat_input.size()[0],concat_input.size()[1],self.layer_width+1) + concat_input.view(concat_input.size()[0],concat_input.size()[1],1)
        # stateful lstm
        # .view(1, self.human_rnn_size, self.human_rnn_size)
        # concat_input.view(concat_input.size()[0],concat_input.size()[1],1)
        # torch.cat((h_prev, c_prev), 0).unfold
        # self.cell = nn.LSTM(input_size=1, hidden_size=self.human_rnn_size,
        #                     num_layers=self.num_layers, batch_first=True) # +1 for concat edge
        #
        # output,(h_curr, c_curr) = self.cell(concat_input.unsqueeze(2),(h_prev, c_prev))

        self.cell = nn.GRU(input_size=self.human_node_embedding_size, hidden_size=self.human_rnn_size,
                           num_layers=self.num_layers, batch_first=False)

        # self.cell = nn.LSTM(input_size=self.human_node_embedding_size, hidden_size=self.human_rnn_size, #self.human_node_embedding_size
        #                     num_layers=self.num_layers, batch_first=False)  # +1 for concat edge
        # output, (h_curr, c_curr) = self.cell(concat_input, (h_prev, c_prev))
        output, h_curr = self.cell(concat_input, h_prev)
        c2 = output.squeeze() #torch.split(output, 1, dim=0)
        # c2 = self.softmax(c2)  # Check if this is necessary
        # c3 = [x.squeeze() for x in c2]
        # out1 = [F.linear(input=x, weight=self.who) for x in c3]
        position_output = F.linear(input=output, weight=self.who, bias=self.out_bias)
        # new_input = F.linear(input=velocity , weight=self.win ,bias=self.in_bias) #final_output +
        # encoded_input = F.linear(input=new_input, weight=torch.Tensor(self.human_node_embedding_size, new_input.size()[1])
        #                             .uniform_(self.lower_bound, self.upper_bound), bias=self.in_bias)
        # encoded_input = self.relu(encoded_input)
        # output, h_curr = self.vel_gru(encoded_input.unsqueeze(0), h_curr)
        output, h_curr = self.vel_gru(velocity.unsqueeze(0), h_curr)
        # output = output.squeeze()
        vel_output = F.linear(input=output, weight=self.who, bias=self.out_bias)
        # vel_output = self.softmax(vel_output) # keep velocity in positive domain
        # distance_output = self.softmax(distance_output)
        # distance_output = self.softmax(distance_output)
        final_output = vel_output + position_output

        return final_output, h_curr, c_curr, self.max_len


class lstm_network(nn.Module):
    ''' the deep recurrent network is a set of stacked layers: (transformation) hidden layer
    followed by non-linear activation function
    16/2/2019: start with node centric graph design: assign lstm to nodes and let all the processing happen there.
    without need for spatial lstm and temporal lstms as found in st-graph (vemula et al.)
    '''

    def __init__(self, args, file_hook=None):
        super(lstm_network, self).__init__()

        self.args = args

        self.learning_rate = args.learning_rate
        self.output_size = args.output_size
        self.humanNode = HumanNode(args, file_hook=file_hook)
        self.num_layers = self.args.num_layers

        self.human_node_hidden_states = Variable(
            torch.zeros(self.num_layers, self.args.max_len, self.args.human_node_rnn_size),
            requires_grad=False)
        self.human_node_cell_states = Variable(
            torch.zeros(self.num_layers, self.args.max_len, self.args.human_node_rnn_size),
            requires_grad=False)

    def forward(self, graph_step, velocity, prev_hidden=None, prev_cell=None, differ=False, retain=True):
        step = graph_step.step
        # node_id , nodes = [v.pos[step] for (k ,v) in graph_step.getNodes()[step].items()]
        edges = graph_step.getEdges()
        nodes = graph_step.getNodes()[step]

        numNodes = len(nodes)
        # numedges = len(edges[0])
        if numNodes > self.args.max_len:
            self.args.max_len = numNodes
            self.humanNode.max_len = self.args.max_len

        if step == 1: #and prev_hidden == None and  prev_cell == None:
            prev_hidden = torch.zeros(self.num_layers, self.args.max_len, self.args.human_node_rnn_size)
            prev_cell = torch.zeros(self.num_layers, self.args.max_len, self.args.human_node_rnn_size)
        else:
            prev_nodes = graph_step.getNodes()[step-batch_size:step]
            for (i, n) in zip(range(len(prev_nodes)), prev_nodes):
                for k in n:
                    if k < len(graph_step.nodes[i]): # and i == (step - 4 - 1):
                        # prev_cell
                        prev_hidden[:,i, :], prev_cell[:, i, :] = \
                            n[k].getState()

        pos, common_nodes , exc_node = cr.get_node_positions(graph_step, step)

        cut_idx = sorted(set(range(len(graph_step.getNodes()[step - 1]))) - set(range(len(exc_node))))
        if len(cut_idx) > self.args.max_len:
            self.args.max_len = len(cut_idx)
            self.humanNode.max_len = self.args.max_len

        if len(edges[step - 1]):
            node_edges = []
            for e in edges[step - batch_size:step]:
                for x in e:
                    # if x.dist not in node_edges:
                    node_edges.append(x.dist)

            node_edges = torch.stack(node_edges, dim=0)

            # node_edges = np.resize(node_edges, (1, len(node_edges)))
            # node_edges = torch.reshape(node_edges, (self.num_layers, round(node_edges.size()[0] / self.num_layers)))
            node_edges = node_edges.float()
            if len(node_edges) < self.args.max_len:
                node_edges = torch.cat((node_edges, torch.zeros(abs(self.args.max_len - len(node_edges)))))

            node_edges = node_edges.unsqueeze(0)
            #torch.zeros(self.num_layers, abs(self.args.max_len - node_edges.size()[1]))), dim=1)
        else:
            node_edges = Variable(torch.ones(1, self.args.max_len))

        # if node_edges.size()[1] > self.args.max_len:
        #     node_edges = node_edges[:,0:self.args.max_len]
        # else:
        #     node_edges = torch.cat((node_edges, torch.zeros((len(node_edges), abs(node_edges.size()[1] - self.args.max_len)))), dim=1)
        # for sdd ped_ids are integers of different range
        # cut_idx = sorted(set(graph_step.getNodes()[step - 1]) - set(exc_node))
        # cut_idx -= np.ones((len(cut_idx)))
        # if np.any(cut_idx > 10):
        #     p = cut_idx != 10
        #     cut_idx[p] =  cut_idx[p] - (cut_idx[p]/10) * 10
        # cut_size = torch.empty(1, -(new_len - old_len), self.human_rnn_size)
        # step-1 for index of edges list beginning from 0 to len-1
        if pos.size()[0] < self.num_layers:
            pos = torch.cat((pos, torch.zeros(
                ((self.num_layers - pos.size()[0]), pos.size()[1], pos.size()[2]))), dim=0)

        if pos.size()[1] < self.args.max_len:
            node_pos = Variable(torch.cat((pos, torch.zeros((self.args.num_layers,(self.args.max_len - pos.size()[1]),
                                                             pos.size()[2]))), dim=1)).requires_grad_(False)

        else:
            node_pos = Variable(pos).requires_grad_(False)

        prev_hidden, prev_cell, disp_len = cr.update_hidden_state_tensor(prev_hidden ,
                                prev_cell ,cut_idx)

        # *************************************************
        # if step == 0:
        graph_step.adj_mat = sg.init_adjecancy_matrix(graph_size= len(nodes),max_size=self.args.max_len ,mode=0)
        # elif step >=3:
            # if step == 0:
            #     self.humanNode.reset_parameters(len(nodes))

        graph_step.dist_mat = torch.zeros(self.args.max_len, self.args.max_len) #torch.zeros(self.args.max_len, self.args.max_len)
        sg.distance_matrix(graph_nodes=pos,dist_mat=graph_step.dist_mat) #, dist_mat=graph_step.dist_mat
        # graph_step.dist_mat = graph_step.dist_mat**-1

        if len(graph_step.dist_mat) < self.args.max_len:
            graph_step.dist_mat = torch.cat((graph_step.dist_mat, torch.zeros(((self.args.max_len - len(pos)), graph_step.dist_mat.size()[1]))))
        # *************************************************
        # node_edges = torch.cat((graph_step.dist_mat, node_edges) , dim=1)
        # outputs_acc = torch.empty_like(node_pos)
        outputs, hidden_state, cell_state, max_len = self.humanNode(position=node_pos,
                                                                    distance=node_edges[:,0:self.args.max_len], velocity=velocity,
                                                                    h_prev=prev_hidden, c_prev=prev_cell)

        # self.args.max_len = max_len
        outputs_acc = outputs[0:len(node_edges),0:node_pos.size()[1]] + node_pos  # make use of consequent predictions(incremental learning)
        # outputs_acc = outputs_acc.unsqueeze(2)

        for i in range(self.args.pred_len-1):
            outputs, hidden_state, cell_state, max_len = self.humanNode(position=outputs[0:len(node_pos),0:node_pos.size()[1]] + node_pos, \
                                    distance=node_edges[:,0:self.args.max_len], velocity= velocity, h_prev=prev_hidden,c_prev=prev_cell,
                                    differ=differ,updateAdj=True, adj_mat=graph_step.adj_mat,dist_mat=graph_step.dist_mat, retain=retain)

            # node_pos = Variable(pos)
            # node_pos += outputs # in-place operation
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # node_pos = Variable(pos).requires_grad_(False)
            # outputs_acc = torch.cat((node_pos, outputs_acc), 1)
            outputs_acc = torch.cat((outputs[0:len(node_edges),0:node_pos.size()[1]]  + node_pos, outputs_acc), 1)

        # TODO: refer hidden states back to their nodes in the main Graph class, for more accurate
        #      indexing of hidden states when number of nodes change, just call each node state and collate the states back in tabulated format
        # outputs_acc = F.avg_pool3d(outputs_acc.unsqueeze(dim=0), kernel_size=(batch_size, 1, 1))

        # Changed to max Pooling to select biggest values in predictions along first axis
        # as a variational approach the gru cell outputs 4 different prediction along every step for every pedestrian
        outputs_acc = outputs_acc[3,:,:] #F.max_pool3d(outputs_acc.unsqueeze(dim=0), kernel_size=(4, 1, 1))
        # outputs_acc = F.max_pool3d(outputs_acc.unsqueeze(dim=0), kernel_size=(4, 1, 1))
        outputs_acc = outputs_acc.squeeze()  # outputs_acc[3,:,:] #torch.sum(outputs_acc, dim=0) #
        out_sh = outputs_acc.size()
        outputs_acc = torch.reshape(outputs_acc, shape=(self.args.pred_len, int(out_sh[0] / self.args.pred_len), 2))

        for (i, n) in zip(range(hidden_state.size()[1]), nodes):  # nodes object have same order as outputs_acc array
            # if i < len(common_nodes):
            if i < outputs_acc.size()[1]:
                graph_step.nodes[step][n].setState(hidden_state[:, i, :], cell_state)  # cell_state[:, i, :]
                graph_step.nodes[step][n].setPrediction(outputs_acc[:, i])  # *self.args.pred_len:i*self.args.pred_len+self.args.pred_len

        if step == 1:
            self.human_node_cell_states = self.human_node_cell_states + cell_state
            self.human_node_hidden_states = self.human_node_hidden_states + hidden_state
        else:
            self.human_node_hidden_states = hidden_state
            self.human_node_cell_states = cell_state
            # self.human_node_hidden_states = torch.cat((self.human_node_hidden_states, hidden_state))
            # self.human_node_cell_states = torch.cat((self.human_node_cell_states, cell_state))

        return outputs_acc, self.human_node_hidden_states, self.human_node_cell_states , max_len, exc_node, disp_len
