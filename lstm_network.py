# '''
#  Class lstm_network to define and intialize lstm based model
# '''
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import functional as F
#
# import criterion as cr
# import math
# '''
# the spatio-temporal lstm network has simplified design choice, by assigning LSTMs only at the node.
# excluding edge lstms from this deign means that the graph is node-centric.
# TODO: we will check the choice of interaction-centric design.
# '''
#
# # TODO: add bias , add new feature to explicitly represent velocity on the end layer when making outputs (new final layer)
# class HumanNode(nn.Module):
#     def __init__(self, args, file_hook=None):
#         super(HumanNode, self).__init__()
#         # self.lstm_size = args.args.lstm_size
#         self.human_rnn_size = args.human_node_rnn_size
#         self.human_node_input_size = args.human_node_input_size
#         self.human_node_output_size = args.human_node_output_size
#         self.human_node_embedding_size = args.human_node_embedding_size
#         self.max_len = args.max_len
#
#         # Change lower/upper bounds to see how it fits wider range of dynamics change rates
#         self.lower_bound = -1 / self.human_node_embedding_size
#         self.upper_bound = 1  / self.human_node_embedding_size #2
#
#         self.relu = nn.PReLU(init=0.0) #nn.ReLU() #
#         self.softmax = nn.Softmax()
#         self.softLog = nn.LogSoftmax(dim=1)
#         self.num_layers = args.num_layers
#
#         # if not np.any([self.win, self.w_edge, self.who, self.who2, self.who3]):
#         self.register_parameter('win', None)
#         self.register_parameter('who', None)
#         self.register_parameter('w_edge', None)
#
#         self.file_hook = file_hook
#         # self.encoding_layer = nn.Linear(self.human_node_input_size , self.human_node_embedding_size)
#         #
#         # self.embeddeing_len_param = nn.Parameter(torch.zeros(1,1)) # parameters are not volatile by default
#         # self.human_node_embedding_param = nn.Parameter(torch.zeros(1,1)+self.human_node_embedding_size)
#
#         # self.dyn_edge_embedding = F.linear(self.embeddeing_len, self.human_node_embedding_param)
#         # self.edge_embedding = nn.Linear(1, self.human_node_embedding_size)
#
#         # self.cell = nn.LSTMCell(self.human_node_embedding_size , self.human_rnn_size)
#         # 2*human_node_embedding_size for concatenated node input(position) and edge input(distance)
#
#     def reset_parameters(self, input):
#         bound = self.lower_bound
#         self.win = nn.Parameter(torch.Tensor(self.human_node_embedding_size, 2)
#                                 .uniform_(self.lower_bound, self.upper_bound).requires_grad_())
#         # bound, _ = nn.init._calculate_fan_in_and_fan_out(self.win)
#         # bound = 1 / math.sqrt(bound)
#         self.in_bias = nn.Parameter(torch.Tensor(1,self.human_node_embedding_size).uniform_(-bound, bound).requires_grad_())
#
#
#         self.w_edge = nn.Parameter(torch.Tensor(self.human_node_embedding_size, input.size()[0])
#                                    .uniform_(self.lower_bound, self.upper_bound)
#                                    .requires_grad_())
#         # bound, _ = nn.init._calculate_fan_in_and_fan_out(self.w_edge)
#         # bound = 1 / math.sqrt(bound)
#         self.ih_bias = nn.Parameter(torch.Tensor(1, self.human_node_embedding_size)
#                                     .uniform_(-bound, bound).requires_grad_())
#
#
#         self.who = nn.Parameter(
#             torch.Tensor(2, self.human_rnn_size).uniform_(self.lower_bound, self.upper_bound).requires_grad_())
#
#         # bound, _ = nn.init._calculate_fan_in_and_fan_out(self.who)
#         # bound = 1 / math.sqrt(bound)
#         self.out_bias = nn.Parameter(torch.Tensor(1,2).uniform_(-bound , bound))
#
#         self.register_parameters()
#
#     def register_parameters(self):
#         self.register_parameter(name='win', param=self.win)
#         self.register_parameter(name='in_bias', param=self.in_bias)
#
#         self.register_parameter(name='who', param=self.who)
#         self.register_parameter(name='out_bias', param=self.out_bias)
#
#         self.register_parameter(name='w_edge', param=self.w_edge)
#         self.register_parameter(name='ih_bias', param=self.ih_bias)
#
#     def reset_dynamic_parameter(self, input):
#         self.w_edge_c = nn.Parameter(self.w_edge[:, 0:len(input)])
#         # self.w_edge = nn.Parameter(torch.Tensor(self.human_node_embedding_size, input.size()[0])
#         #                            .uniform_(self.lower_bound, self.upper_bound).requires_grad_())
#         self.register_parameter('w_edge_weight', self.w_edge_c)
#
#     def forward(self, position, distance, h_prev, c_prev, differ=False):
#         # encoded_input = self.encoding_layer(position)
#         self.layer_width = position.size()[0]
#         c_curr = torch.zeros(1,256)
#
#         # .uniform_(self.lower_bound,self.upper_bound)
#         x = [self.win, self.w_edge, self.who] #, self.who2, self.who3
#         y = None
#         if np.all([a is None for a in x]):
#             self.reset_parameters(input=distance)
#             self.w_edge_c = self.w_edge
#
#         if differ:
#             # self.w_edge = self.w_edge[:, 0:len(distance)
#             self.reset_dynamic_parameter(input=distance)
#
#         # used the functional linear transformation from nn.functional
#         # as we are expecting a variable input length
#         encoded_input = F.linear(input=position, weight=self.win, bias=self.in_bias)
#
#         encoded_input = self.relu(encoded_input)
#
#         embedded_edge = F.linear(input=distance.t(),
#                                  weight=self.w_edge_c, bias=self.ih_bias)  # randn needs to be bounded by smaller interval
#         # embedded_edge = embedded_edge.unsqueeze(dim=0)
#         # self.edge_embedding = nn.Linear(distance.size()[1], self.human_node_embedding_size)
#         # embedded_edge = self.edge_embedding(distance)
#         embedded_edge = self.relu(embedded_edge)  # relu with randn generate higher sparsity matrix, use prelu better
#
#         # TODO: layer for embedding spatial edges (possibly pooling layer ?) by 15th march
#         # TODO: learnable criterion for growing spatial edges(growing spatial hidden feature vectors in width and depth)
#         # by 10th March xxx by 15th march
#         # concat_input = torch.cat((encoded_input, embedded_edge), 0)
#         # concat_input = torch.cat((encoded_input,embedded_edge),1) + torch.zeros((h_prev.size()))
#
#         concat_input = torch.cat((encoded_input, embedded_edge), 1)
#         concat_input = concat_input.unsqueeze(dim=0)
#         if concat_input.size()[1] > self.max_len:
#             concat_input = torch.cat((concat_input, torch.zeros((1, (concat_input.size()[1] - self.max_len ), self.human_rnn_size))))
#             self.max_len = concat_input.size()[1]
#
#         self.cell = nn.GRU(input_size=self.human_rnn_size, hidden_size=self.human_rnn_size,
#                             num_layers=self.num_layers, batch_first=False)  # +1 for concat edge
#         # self.cell = nn.LSTM(input_size=self.human_node_embedding_size, hidden_size=self.human_rnn_size,
#         #                num_layers=self.num_layers, batch_first=True)
#         # output, (h_curr, c_curr) = self.cell(concat_input, (h_prev, c_prev))
#         output, h_curr = self.cell(concat_input, h_prev)
#
#
#         c2 = output.squeeze()  # torch.split(output, 1, dim=0)
#         c2 = self.softmax(c2)  # Check if this is necessary
#         # c3 = [x.squeeze() for x in c2]
#         # out1 = [F.linear(input=x, weight=self.who) for x in c3]
#         final_output = F.linear(input=c2, weight=self.who, bias=self.out_bias)
#         # out2 = [F.linear(input=x.t(), weight=self.who2) for x in out1]
#         # out2 = self.relu(torch.stack(out2))
#         # out2 = self.relu(torch.stack(out1))
#
#         # self.bias = torch.Tensor(1, 2).uniform_(-1/256, 1/256).requires_grad_()
#         # final_output = [F.linear(input=x, weight=self.who3) for x in out2]
#         # final_output = torch.stack(final_output).squeeze()
#         # final_output = self.relu(final_output)
#         # output = self.output_layer(h_curr) # linear transformation
#         # self.who = torch.Tensor(1, self.human_rnn_size).normal_().requires_grad_()
#         # c2 = torch.split(output, 1, dim=0)
#         # c3 = [x.squeeze() for x in c2]
#         # out1 = [F.linear(input=x, weight=self.who) for x in c3]
#         #
#         # self.who2 = torch.Tensor(1, self.human_rnn_size).normal_().requires_grad_()
#         # out2 = [F.linear(input=x.t(), weight=self.who2) for x in out1]
#         #
#         # self.who3 = torch.Tensor(2, 1).normal_().requires_grad_()
#         # # self.bias = torch.Tensor(1, 2).uniform_(-1/256, 1/256).requires_grad_()
#         # out3 = [F.linear(input=x, weight=self.who3) for x in out2]
#
#         # self.file_hook.write('out1 \n' + str(out1) + '\n out2 \n' + str(out2) + '\n final output \n' + str(final_output))
#         final_output = self.softmax(final_output)
#         return final_output, h_curr, c_curr, self.max_len
#
#
# class lstm_network(nn.Module):
#     ''' the deep recurrent network is a set of stacked layers: (transformation) hidden layer
#     followed by non-linear activation function
#     16/2/2019: start with node centric graph design: assign lstm to nodes and let all the processing happen there.
#     without need for spatial lstm and temporal lstms as found in st-graph (vemula et al.)
#     '''
#
#     def __init__(self, args, file_hook=None):
#         super(lstm_network, self).__init__()
#
#         self.args = args
#
#         self.learning_rate = args.learning_rate
#         self.output_size = args.output_size
#         self.humanNode = HumanNode(args, file_hook=file_hook)
#         self.num_layers = self.args.num_layers
#
#         self.human_node_hidden_states = Variable(torch.zeros(self.args.max_len, self.args.human_node_rnn_size),
#                                                  requires_grad=True)
#         self.human_node_cell_states = Variable(torch.zeros(self.args.max_len, self.args.human_node_rnn_size),
#                                                requires_grad=True)
#
#     def forward(self, graph_step, prev_hidden=None, prev_cell=None, differ=False):
#         step = graph_step.step
#         # node_id , nodes = [v.pos[step] for (k ,v) in graph_step.getNodes()[step].items()]
#         edges = graph_step.getEdges()
#         nodes = graph_step.getNodes()[step]
#         # node_edges = []
#
#         numNodes = len(nodes)
#         # numedges = len(edges[0])
#         if step == 1:#or prev_hidden == None and prev_cell == None:
#             prev_hidden = torch.zeros(self.num_layers, numNodes, self.args.human_node_rnn_size)
#             prev_cell = torch.zeros(self.num_layers, numNodes, self.args.human_node_rnn_size)
#         else:
#             prev_nodes = graph_step.getNodes()[step-4:step]
#             for (i, n) in zip(range(len(prev_nodes)), prev_nodes):
#                 for k in n:
#                     if k < len(graph_step.nodes[i]):
#                         # prev_cell[:, i, :]
#                         prev_hidden[:,i, :], prev_cell = graph_step.nodes[i][k].getState()
#         # outputs_acc = Variable(torch.zeros(numNodes, self.output_size))
#         # pos = []
#         pos, common_nodes, exc_node = cr.get_node_positions(graph_step, step)
#         # for i, (k,v) in zip(range(len(nodes)), nodes):
#         #     pos.append((v.pos[step]))
#         # edge_id = (k, k)
#         if len(edges[step - 1]):
#             # print(len(edges))
#             # [print (e) for e in edges[step-1]]
#             node_edges = []
#             for e in edges[step - 4:step]:
#                 for x in e:
#                     if x.dist not in node_edges:
#                         node_edges.append(x.dist)
#
#             node_edges = torch.stack(node_edges, dim=0)
#             # node_edges = np.resize(node_edges, (1, len(node_edges)))
#             node_edges = node_edges.float().unsqueeze(0)
#             if node_edges.size()[1] < self.args.max_len:
#                 node_edges = Variable(torch.cat((node_edges, torch.zeros((1, (self.args.max_len - node_edges.size()[1]))
#                                     )), 1)).requires_grad_(False)
#         else:
#             node_edges = Variable(torch.zeros(1, self.args.max_len))
#         # for sdd ped_ids are integers of different range
#         cut_idx = sorted(set(range(len(graph_step.getNodes()[step - 1]))) - set(range(len(exc_node))))
#         # cut_idx = sorted(set(graph_step.getNodes()[step - 1]) - set(exc_node))
#         # cut_idx -= np.ones((len(cut_idx)))
#         # if np.any(cut_idx > 10):
#         #     p = cut_idx != 10
#         #     cut_idx[p] =  cut_idx[p] - (cut_idx[p]/10) * 10
#         # cut_size = torch.empty(1, -(new_len - old_len), self.human_rnn_size)
#         # step-1 for index of edges list beginning from 0 to len-1
#         if len(pos) < self.args.max_len:
#             node_pos = Variable(torch.cat((pos, torch.zeros(((self.args.max_len - len(pos)), pos.size()[1])))) \
#                                 ).requires_grad_(False)
#         else:
#             node_pos = Variable(pos).requires_grad_(False)
#
#         prev_hidden, prev_cell, disp_len = cr.update_hidden_state_tensor(prev_hidden,
#                                                                          prev_cell, cut_idx)
#         # outputs_acc = torch.empty_like(node_pos)
#         # shuffle = torch.randperm(node_edges.size()[1])
#         # node_edges = node_edges[:, shuffle]
#         # node_pos = node_pos[shuffle,:]
#         outputs, hidden_state, cell_state, max_len = self.humanNode(position=node_pos, \
#                                                                     distance=node_edges, h_prev=prev_hidden,
#                                                                     c_prev=prev_cell)
#
#         outputs_acc = outputs + node_pos  # make use of consequent predictions(incremental learning)
#         # outputs_acc = outputs_acc.unsqueeze(2)
#         for i in range(self.args.pred_len - 1):
#             outputs, hidden_state, cell_state, max_len = self.humanNode(position=outputs + node_pos, \
#                                                                         distance=node_edges, h_prev=prev_hidden,
#                                                                         c_prev=prev_cell, differ=differ)
#             # node_pos = Variable(pos)
#             # node_pos += outputs # in-place operation
#             # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
#             # node_pos = Variable(pos).requires_grad_(False)
#             # outputs_acc = torch.cat((node_pos, outputs_acc), 1)
#             outputs_acc = torch.cat((outputs + node_pos, outputs_acc), 1)
#
#         # TODO: refer hidden states back to their nodes in the main Graph class, for more accurate
#         #      indexing of hidden states when number of nodes change, just call each node state and collate the states back in tabulated format
#
#         for (i, n) in zip(range(hidden_state.size()[1]), nodes):  # nodes object have same order as outputs_acc array
#             # if i < len(common_nodes):
#             graph_step.nodes[step][n].setState(hidden_state[:, i, :], cell_state) #cell_state[:, i, :]
#             graph_step.nodes[step][n].setPrediction(outputs_acc[i])
#
#         if step == 1:
#             self.human_node_cell_states = self.human_node_cell_states + cell_state
#             self.human_node_hidden_states = self.human_node_hidden_states + hidden_state
#         else:
#             self.human_node_hidden_states = hidden_state
#             self.human_node_cell_states = cell_state
#             # self.human_node_hidden_states = torch.cat((self.human_node_hidden_states, hidden_state))
#             # self.human_node_cell_states = torch.cat((self.human_node_cell_states, cell_state))
#
#         return outputs_acc, self.human_node_hidden_states, self.human_node_cell_states, max_len, exc_node, disp_len
# ''' Function that implements loss for online prediction
# squared loss ?? tends to maximize error due to using squares .. can this be beneficial for double penalizing
# the estimation error at current steps therefore lowering errors exponentially at a constant rate
# so easier training ?
# '''
#
# import numpy as np
# import torch
# import torch.nn as nn
#
#
# def online_BP(outputs, target):
#     out_sh = outputs.size()
#
#     # outputs = torch.reshape(outputs, shape=(out_sh[0], int(out_sh[1]/2) ,2))
#     out_sh = outputs.size()
#     end = out_sh[1] - 1  # for 1x12 sequence
#     end_tg = target.size()[0] - 1  # for nx12 sequence
#     end_pred = outputs.size()[0] - 1
#     # inf: make lr smaller/ divide by zero
#     # nan: make lr smaller/ sqrt(negative value)
#     # watch over final layer outputs if they explode in value
#     # permute target sequences to be same order of axes as predicted sequences [nx12x2]
#     # target = target.permute(1,0,2)
#     # define loss function
#     loss = nn.MSELoss()  # size_average is by default true , n = number of present nodes in current frame
#     err = loss(outputs[0:len(target)], target)  # divide by (1d*2d*3d) not by batch_size
#
#     norm_ade = torch.norm((outputs[0:len(target)] - target), p=2)  # / out_sh[0]
#     # norm_ade = torch.sqrt(torch.pow(torch.sum(outputs - target), 2.0)) #/ (out_sh[0] * out_sh[1])
#     if out_sh[0] > 1:
#         norm_fde = torch.norm((outputs[end_tg] - target[end_tg]), p=2.0)
#         # torch.sqrt(torch.pow(torch.sum(outputs[end_tg] - target[end_tg])
#         # , 2.0)) # / (out_sh[0])
#     else:
#         norm_fde = torch.norm((outputs[:, end] - target[:, end]), p=2.0)
#         # torch.sqrt(torch.pow(torch.sum(outputs[:, end] - target[:,end])
#         #                                 , 2.0))
#         # print("predicted trajectory" , outputs)
#         # print("\n \n ground-truth ", targets)
#     # TODO: check the RMSE and see how square rooting error affect the learning convergence
#     # err = torch.sqrt(loss(outputs, targets))
#
#     return err, norm_ade, norm_fde
#
#
# def get_node_positions(graph_step, step=0, max_len=20, curr_graph_nodes=None, istargets=False):
#     exc_node = []
#     if not istargets:
#         common_nodes, _, exc_node = get_common_nodes(curr_graph=graph_step, frame=step)
#           # sorted()
#         # doesnt work with some ped_id that are not organized by pattern sorted()
#
#         # pos = []  # np.empty(shape=(dim0, 2), dtype=np.float32)
#
#         if len(common_nodes) == 0:
#             nodes = graph_step.getNodes()[step]
#             pos = torch.zeros((4, len(nodes), 2))
#             for i in range(len(nodes)):
#                 # pos.append([])
#                 for (k, v) in nodes.items():
#                     if len(v.targets) > step:
#                         pos[i] = np.array(v.targets[i][step - 4:step])
#                     else:
#                         pos[i] = np.array(v.targets[i][len(v.targets) - 4:len(v.targets)])
#         else:
#             nodes = graph_step.getNodes()[step - 4:step]
#             pos = []
#             for i in range(len(nodes)):
#                 pos.append([])
#                 for (k, v) in nodes[i].items():
#                     if k in common_nodes:
#                         idx = i
#                         t_idx = step
#                         if len(v.targets):
#                             if i >= len(v.targets):
#                                 idx = len(v.targets) - 1
#                             if step >= len(v.targets[idx]):
#                                 t_idx = len(v.targets)
#
#                             pos[i] = np.array(v.targets[idx][t_idx - 4:t_idx])
#
#         for i in range(len(pos)):
#             if len(pos[i]) < len(common_nodes):
#                 l = np.zeros(((len(common_nodes) - len(pos[i])),2))
#                 for j in range((len(common_nodes) - len(pos[i]))):
#                     l[j] = [0, 0]
#                 pos[i] = np.append(pos[i] , l, axis=0)
#
#     else:
#         common_nodes, _, _ = get_common_nodes(curr_graph=graph_step, frame=step,
#                                     curr_graph_nodes=curr_graph_nodes.getNodes())
#
#         # pos = np.empty(shape=(len(common_nodes), len(graph_step[0]),2), dtype=np.float32)
#         pos = {} # TODO: fix memory assignement
#
#         if step == 0:
#             step += 1
#         # when using minibatch setting step > len(common_nodes)
#         if step >= len(common_nodes):
#             step = len(common_nodes)-1
#
#         if len(common_nodes) and len(common_nodes[step]):
#             for i in range(step , len(graph_step)):
#
#                 nodes = []
#                 for k, x in enumerate(graph_step[i]):
#                     nodes.append((list(x.keys())[0] , k))
#                 for k,x in enumerate(nodes):
#                     # append 12 steps
#                     if k >= max_len: #or i >= len(common_nodes):
#                         break
#                     elif x[0] in common_nodes[step]: #in common_nodes[i]:
#                         try:
#                            pos[x[0]].append(graph_step[i][k][x[0]])
#                         except KeyError:
#                             if x[0] in pos or len(pos):
#                                 pos[x[0]] = [graph_step[i][k][x[0]]]
#                             else:
#                                 pos = {x[0]: [ graph_step[i][k][x[0]] ]}
#
#                         if k < len(curr_graph_nodes.nodes):
#                             curr_graph_nodes.nodes[k][float(x[0])].setTargets(seq=pos[x[0]])
#
#         return pos, common_nodes
#
#     if isinstance(pos, list):
#         pos = np.stack(pos)
#         if np.ndim(pos) > 3:
#             pos = pos.squeeze(2)
#             # pos = torch.Tensor(pos).permute(1, 0, 2)
#         else:
#             pos = pos.squeeze()
#
#         pos = torch.Tensor(np.asfarray(pos))
#
#     return pos , common_nodes,exc_node
#
#
# def get_common_nodes(curr_graph, frame, curr_graph_nodes=None):
#     if frame == 0:
#         frame += 1
#     exc_node = []
#
#     if not isinstance(curr_graph, list):
#         n1 = curr_graph.getNodes()  # new graph
#
#         if frame >= len(n1):
#             frame = len(n1) - 1
#         common_nodes = set(n1[frame]) & set(n1[frame - 1])  # , reverse=False #sorted()
#         l1 = set(n1[frame]) - set(n1[frame - 1])
#         exc_node = sorted(l1 if len(l1) else set(n1[frame - 1]) - set(n1[frame]), reverse=False)
#         return common_nodes, n1, exc_node
#     else:  # for targets
#         n1 = []  # TODO: fix memory assignment
#
#         common_nodes = set(curr_graph_nodes[frame])
#         common_list = []
#         for i in range(len(curr_graph)):
#             n1.append(np.zeros(shape=(len(curr_graph[i])), dtype=np.float32))
#             for k, x in enumerate(curr_graph[i]):
#                 n1[i][k] = list(x.keys())[0]
#
#             inter = set(n1[i]) & set(common_nodes)
#
#             if len(inter):
#                 common_list.append(sorted(inter))
#
#     # l1 = set(n1[frame]) - set(n1[frame - 1])
#     # exc_node = l1 if len(l1) else set(n1[frame-1]) - set(n1[frame])# sorted, reverse=False)
#     # common_list = list(filter(None, common_list))
#     return common_list, n1, exc_node
#
#
# def update_hidden_state_tensor(prev_hidden, prev_cell, cut_idx, max_len=20):
#     old_len = prev_hidden.size()[0]
#     if max_len < old_len:
#         max_len = old_len
#     cat_len = len(cut_idx) - max_len
#
#     # if cat_len > 0:
#     #     prev_hidden = torch.cat((prev_hidden, torch.zeros(1, cat_len, prev_hidden.size()[2])), dim=1)
#     #     prev_cell = torch.cat((prev_cell, torch.zeros(1, cat_len, prev_hidden.size()[2])), dim=1)
#     if cat_len or (old_len < max_len):
#         # if prev_hidden.size()[1] < max_len
#         prev_hidden = torch.index_select(prev_hidden, index=torch.LongTensor(cut_idx[0:max_len]), dim=1)
#         # prev_cell = torch.index_select(prev_cell, index=torch.LongTensor(cut_idx[0:max_len]), dim=1)
#         prev_hidden = torch.cat((prev_hidden, torch.zeros(len(prev_hidden), abs(cat_len), prev_hidden.size()[2])), dim=1)
#         # prev_cell = torch.cat((prev_cell, torch.zeros(1, abs(cat_len), prev_hidden.size()[2])), dim=1)
#
#         # elif cat_len < 0:
#         #     cat_len *= -1
#         #     prev_hidden = torch.cat((prev_hidden, torch.zeros(1, cat_len, prev_hidden.size()[2])), dim=1)
#         #     prev_cell = torch.cat((prev_cell, torch.zeros(1, cat_len, prev_hidden.size()[2])), dim=1)
#
#     return prev_hidden, prev_cell, cat_len

'''
 Class lstm_network to define and intialize lstm based model
'''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import criterion as cr
import math
import time
''' 
the spatio-temporal lstm network has simplified design choice, by assigning LSTMs only at the node.
excluding edge lstms from this deign means that the graph is node-centric. 
TODO: we will check the choice of interaction-centric design.
'''

# TODO: add bias , add new feature to explicitly represent velocity on the end layer when making outputs (new final layer)
class HumanNode(nn.Module):
    def __init__(self, args, file_hook=None):
        super(HumanNode, self).__init__()
        # self.lstm_size = args.args.lstm_size
        self.human_rnn_size = args.human_node_rnn_size
        self.human_node_input_size = args.human_node_input_size
        self.human_node_output_size = args.human_node_output_size
        self.human_node_embedding_size = args.human_node_embedding_size
        self.max_len = args.max_len

        # Change lower/upper bounds to see how it fits wider range of dynamics change rates
        self.lower_bound = -1 / self.human_node_embedding_size
        self.upper_bound = 1 / self.human_node_embedding_size #2

        self.relu = nn.PReLU(init=0.0) #nn.ReLU() #
        self.softmax = nn.Softmax()
        self.softLog = nn.LogSoftmax(dim=1)
        self.num_layers = args.num_layers

        self.vel_gru = nn.GRU(input_size=self.human_node_embedding_size, hidden_size=self.human_rnn_size,
                              num_layers=self.num_layers, batch_first=False)

        # if not np.any([self.win, self.w_edge, self.who, self.who2, self.who3]):
        self.register_parameter('win', None)
        self.register_parameter('who', None)
        self.register_parameter('w_edge', None)

        self.file_hook = file_hook
        # self.encoding_layer = nn.Linear(self.human_node_input_size , self.human_node_embedding_size)
        #
        # self.embeddeing_len_param = nn.Parameter(torch.zeros(1,1)) # parameters are not volatile by default
        # self.human_node_embedding_param = nn.Parameter(torch.zeros(1,1)+self.human_node_embedding_size)

        # self.dyn_edge_embedding = F.linear(self.embeddeing_len, self.human_node_embedding_param)
        # self.edge_embedding = nn.Linear(1, self.human_node_embedding_size)

        # self.cell = nn.LSTMCell(self.human_node_embedding_size , self.human_rnn_size)
        # 2*human_node_embedding_size for concatenated node input(position) and edge input(distance)

    def reset_parameters(self, input):
        bound = self.lower_bound
        self.win = nn.Parameter(torch.Tensor(self.human_node_embedding_size, 2)
                                .uniform_(self.lower_bound, self.upper_bound).requires_grad_())
        # bound, _ = nn.init._calculate_fan_in_and_fan_out(self.win)
        # bound = 1 / math.sqrt(bound)
        self.in_bias = nn.Parameter(torch.Tensor(1,self.human_node_embedding_size).uniform_(-bound, bound).requires_grad_())


        self.w_edge = nn.Parameter(torch.Tensor(self.human_node_embedding_size, input.size()[0])
                                   .uniform_(self.lower_bound, self.upper_bound)
                                   .requires_grad_())
        # bound, _ = nn.init._calculate_fan_in_and_fan_out(self.w_edge)
        # bound = 1 / math.sqrt(bound)
        self.ih_bias = nn.Parameter(torch.Tensor(1, self.human_node_embedding_size)
                                    .uniform_(-bound, bound).requires_grad_())


        self.who = nn.Parameter(
            torch.Tensor(2, self.human_rnn_size).uniform_(self.lower_bound, self.upper_bound).requires_grad_())

        # bound, _ = nn.init._calculate_fan_in_and_fan_out(self.who)
        # bound = 1 / math.sqrt(bound)
        self.out_bias = nn.Parameter(torch.Tensor(1,2).uniform_(-bound , bound))

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
        # self.w_edge = nn.Parameter(torch.Tensor(self.human_node_embedding_size, input.size()[0])
        #                            .uniform_(self.lower_bound, self.upper_bound).requires_grad_())
        self.register_parameter('w_edge_weight', self.w_edge_c)

    def forward(self, position, distance, velocity, h_prev, c_prev, differ=False):
        # encoded_input = self.encoding_layer(position)
        self.layer_width = position.size()[0]
        c_curr = torch.zeros(1,256)

        # .uniform_(self.lower_bound,self.upper_bound)
        x = [self.win, self.w_edge, self.who] #, self.who2, self.who3
        y = None
        if np.all([a is None for a in x]):
            self.reset_parameters(input=distance)
            self.w_edge_c = self.w_edge

        if differ:
            # self.w_edge = self.w_edge[:, 0:len(distance)
            self.reset_dynamic_parameter(input=distance)

        # used the functional linear transformation from nn.functional
        # as we are expecting a variable input length
        encoded_input = F.linear(input=position, weight=self.win, bias=self.in_bias)

        encoded_input = self.relu(encoded_input)

        embedded_edge = F.linear(input=distance.t(),
                                 weight=self.w_edge_c, bias=self.ih_bias)  # randn needs to be bounded by smaller interval
        # embedded_edge = embedded_edge.unsqueeze(dim=0)
        # self.edge_embedding = nn.Linear(distance.size()[1], self.human_node_embedding_size)
        # embedded_edge = self.edge_embedding(distance)
        embedded_edge = self.relu(embedded_edge)  # relu with randn generate higher sparsity matrix, use prelu better

        # TODO: layer for embedding spatial edges (possibly pooling layer ?) by 15th march
        # TODO: learnable criterion for growing spatial edges(growing spatial hidden feature vectors in width and depth)
        # by 10th March xxx by 15th march
        # concat_input = torch.cat((encoded_input, embedded_edge), 0)
        # concat_input = torch.cat((encoded_input,embedded_edge),1) + torch.zeros((h_prev.size()))
        embedded_edge = embedded_edge.unsqueeze(0)
        concat_input = torch.cat((encoded_input, embedded_edge), 0)
        # concat_input = concat_input.unsqueeze(dim=0)
        if concat_input.size()[1] > self.max_len:
            concat_input = torch.cat((concat_input, torch.zeros((len(concat_input), (concat_input.size()[1] - self.max_len), self.human_node_embedding_size))), dim=0)
            # self.human_rnn_size
            self.max_len = concat_input.size()[1]

        self.cell = nn.GRU(input_size=self.human_node_embedding_size, hidden_size=self.human_rnn_size,
                            num_layers=self.num_layers, batch_first=False)  # +1 for concat edge
        # self.cell = nn.LSTM(input_size=self.human_node_embedding_size, hidden_size=self.human_rnn_size,
        #                num_layers=self.num_layers, batch_first=True)
        # output, (h_curr, c_curr) = self.cell(concat_input, (h_prev, c_prev))
        output, h_curr = self.cell(concat_input, h_prev)

        c2 = output.squeeze()  # torch.split(output, 1, dim=0)
        # c2 = self.softmax(c2)  # Check if this is necessary
        # c3 = [x.squeeze() for x in c2]
        # out1 = [F.linear(input=x, weight=self.who) for x in c3]
        final_output = F.linear(input=c2, weight=self.who, bias=self.out_bias)
        # out2 = [F.linear(input=x.t(), weight=self.who2) for x in out1]
        # out2 = self.relu(torch.stack(out2))
        # out2 = self.relu(torch.stack(out1))
        # self.bias = torch.Tensor(1, 2).uniform_(-1/256, 1/256).requires_grad_()
        # final_output = [F.linear(input=x, weight=self.who3) for x in out2]
        # final_output = torch.stack(final_output).squeeze()
        # final_output = self.relu(final_output)
        # output = self.output_layer(h_curr) # linear transformation
        # self.who = torch.Tensor(1, self.human_rnn_size).normal_().requires_grad_()
        # c2 = torch.split(output, 1, dim=0)
        # c3 = [x.squeeze() for x in c2]
        # out1 = [F.linear(input=x, weight=self.who) for x in c3]
        #
        # self.who2 = torch.Tensor(1, self.human_rnn_size).normal_().requires_grad_()
        # out2 = [F.linear(input=x.t(), weight=self.who2) for x in out1]
        #
        # self.who3 = torch.Tensor(2, 1).normal_().requires_grad_()
        # # self.bias = torch.Tensor(1, 2).uniform_(-1/256, 1/256).requires_grad_()
        # out3 = [F.linear(input=x, weight=self.who3) for x in out2]
        # self.file_hook.write('out1 \n' + str(out1) + '\n out2 \n' + str(out2) + '\n final output \n' + str(final_output))
        # final_output = self.softmax(final_output)

        new_input = final_output + velocity
        encoded_input = F.linear(input=new_input, weight=self.win, bias=self.in_bias)
        encoded_input = self.relu(encoded_input)
        # encoded_input = encoded_input.unsqueeze(dim=0)

        output, h_curr = self.vel_gru(encoded_input, h_curr)

        output = output.squeeze()
        final_output = F.linear(input=output, weight=self.who, bias=self.out_bias)

        return final_output, h_curr, c_curr, self.max_len

    # def forward (self, outputs ,velocity , hidden_state=None):
    #     new_input = outputs+velocity
    #     encoded_input = F.linear(input=new_input, weight=self.win, bias=self.in_bias)
    #     encoded_input = self.relu(encoded_input)
    #     encoded_input = encoded_input.unsqueeze(dim=0)
    #
    #     outputs, hidden_state = self.vel_gru(encoded_input , hidden_state)
    #     return outputs, hidden_state

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

        self.human_node_hidden_states = Variable(torch.zeros(self.num_layers,self.args.max_len, self.args.human_node_rnn_size),
                                                 requires_grad=False)
        self.human_node_cell_states = Variable(torch.zeros(self.num_layers, self.args.max_len, self.args.human_node_rnn_size),
                                               requires_grad=False)


    def forward(self, graph_step, velocity ,prev_hidden=None, prev_cell=None, differ=False):
        step = graph_step.step
        # node_id , nodes = [v.pos[step] for (k ,v) in graph_step.getNodes()[step].items()]
        edges = graph_step.getEdges()
        nodes = graph_step.getNodes()[step]
        # node_edges = []

        numNodes = len(nodes)
        # numedges = len(edges[0])
        # if step == 1:#or prev_hidden == None and prev_cell == None:
        #     prev_hidden = torch.zeros(self.num_layers, numNodes, self.args.human_node_rnn_size)
        #     prev_cell = torch.zeros(self.num_layers, numNodes, self.args.human_node_rnn_size)
        # else:
        prev_nodes = graph_step.getNodes()[step-4:step]
        for (i, n) in zip(range(len(prev_nodes)), prev_nodes):
            for k in n:
                if k < len(graph_step.nodes[i]):
                    # prev_cell[:, i, :]
                    prev_hidden[:,i, :], prev_cell = graph_step.nodes[i][k].getState()

        pos, common_nodes, exc_node = cr.get_node_positions(graph_step, step)
        # pos = pos.permute(1,0,2)
        # for i, (k,v) in zip(range(len(nodes)), nodes):
        #     pos.append((v.pos[step]))
        # edge_id = (k, k)
        if len(edges[step - 1]):
            # print(len(edges))
            # [print (e) for e in edges[step-1]]
            node_edges = []
            for e in edges[step - 4:step]:
                for x in e:
                    if x.dist not in node_edges:
                        node_edges.append(x.dist)

            node_edges = torch.stack(node_edges, dim=0)
            # node_edges = np.resize(node_edges, (1, len(node_edges)))
            node_edges = node_edges.float().unsqueeze(0)
            if node_edges.size()[1] < self.args.max_len:
                node_edges = Variable(torch.cat((node_edges, torch.zeros((1, (self.args.max_len - node_edges.size()[1]))
                                    )), 1)).requires_grad_(False)
        else:
            node_edges = Variable(torch.zeros(1, self.args.max_len))

        # for sdd ped_ids are integers of different range
        cut_idx = sorted(set(range(len(graph_step.getNodes()[step - 1]))) - set(range(len(exc_node))))

        if pos.size()[1] < self.args.max_len:
            node_pos = Variable(torch.cat((pos, torch.zeros((self.args.num_layers,
                                (self.args.max_len - pos.size()[1]), pos.size()[2]))), dim=1 )).requires_grad_(False)
        else:
            node_pos = Variable(pos).requires_grad_(False)

        prev_hidden, prev_cell, disp_len = cr.update_hidden_state_tensor(prev_hidden,
                                                             prev_cell, cut_idx, max_len=self.args.max_len)

        outputs, hidden_state, cell_state, max_len = self.humanNode(position=node_pos,
                                                                    distance=node_edges[:, 0:self.args.max_len],
                                                                    velocity=velocity,
                                                                    h_prev=prev_hidden, c_prev=prev_cell)

        # self.args.max_len = max_len
        outputs_acc = outputs[0:len(node_edges),0:node_pos.size()[1]] + node_pos  # make use of consequent predictions(incremental learning)
        # outputs_acc = outputs_acc.unsqueeze(2)

        for i in range(self.args.pred_len - 1):
            outputs, hidden_state, cell_state, max_len = self.humanNode(
                position=outputs[0:len(node_edges), 0:node_pos.size()[1]] + node_pos, \
                distance=node_edges[:, 0:self.args.max_len], velocity=velocity, h_prev=prev_hidden, c_prev=prev_cell,
                differ=differ)


            outputs_acc = torch.cat((outputs[0:len(node_edges), 0:node_pos.size()[1]] + node_pos, outputs_acc), 1)

        # TODO: refer hidden states back to their nodes in the main Graph class, for more accurate
        #      indexing of hidden states when number of nodes change, just call each node state and collate the states back in tabulated format
        start_pool = time.time()
        outputs_acc = F.avg_pool3d(outputs_acc.unsqueeze(dim=0), kernel_size=(4,1,1))
        end_pool = time.time()
        print('average pooling took: ', (end_pool - start_pool), ' seconds')

        outputs_acc = outputs_acc.squeeze() #outputs_acc[3,:,:] #torch.sum(outputs_acc, dim=0) #
        out_sh = outputs_acc.size()
        outputs_acc = torch.reshape(outputs_acc, shape=(self.args.pred_len,int(out_sh[0] / self.args.pred_len), 2))

        for (i, n) in zip(range(hidden_state.size()[1]), nodes):  # nodes object have same order as outputs_acc array
            # if i < len(common_nodes):
            graph_step.nodes[step][n].setState(hidden_state[:, i, :], cell_state) #cell_state[:, i, :]
            graph_step.nodes[step][n].setPrediction(outputs_acc[:,i])#*self.args.pred_len:i*self.args.pred_len+self.args.pred_len

        if step == 1:
            self.human_node_cell_states = self.human_node_cell_states + cell_state
            self.human_node_hidden_states = self.human_node_hidden_states + hidden_state
        else:
            self.human_node_hidden_states = hidden_state
            self.human_node_cell_states = cell_state
            # self.human_node_hidden_states = torch.cat((self.human_node_hidden_states, hidden_state))
            # self.human_node_cell_states = torch.cat((self.human_node_cell_states, cell_state))

        return outputs_acc, self.human_node_hidden_states, self.human_node_cell_states, max_len, exc_node, disp_len
