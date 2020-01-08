''' Function that implements loss for online prediction
squared loss ?? tends to maximize error due to using squares .. can this be beneficial for double penalizing
the estimation error at current steps therefore lowering errors exponentially at a constant rate
so easier training
'''

import torch
import numpy as np
import torch.nn as nn
batch_size = 8
def online_BP(outputs, target):

    out_sh = outputs.size()
    end = out_sh[0] - 1 # for 1x12 sequence
    end_tg = target.size()[0] - 1 # for nx12 sequence

    # define loss function
    loss = nn.MSELoss()  # size_average is by default true , n = number of present nodes in current frame
    err = loss(outputs[0:len(target)], target) # divide by (1d*2d*3d) not by batch_size

    norm_ade = (outputs[0:len(target)] - target).pow(2) #torch.norm(input=(outputs[0:len(target)] - target))

    if out_sh[0] > 1:
        # norm_fde = torch.norm(input=(outputs[end]- target[end_tg]))
        norm_fde = outputs[end] - target[end_tg]
        norm_fde = norm_fde.pow(2)

    else:
        norm_fde = (outputs[:, end_tg] - target[:, end_tg]).pow(2) #torch.norm(input=((outputs[:, end_tg]- target[:, end_tg])))

    return err, norm_ade, norm_fde

def get_node_positions(graph_step, pos={}, nodes=[], step=0, max_len=40,curr_graph_nodes=None, istargets=False):
    exc_node = []
    if not istargets:
        common_nodes, _, exc_node = get_common_nodes(curr_graph=graph_step, frame=step)
        # pos = torch.zeros((len(nodes), batch_size, 2))
        if len(common_nodes) == 0:
            for i in range(len(nodes)):
                # pos.append([])
                for (k, v) in nodes.items():
                    if len(v.targets) > step:
                        pos[i] = torch.Tensor(v.targets[step - 4:step])
                    elif len(v.targets):
                        pos[i] = torch.Tensor(v.targets[0:step]) #len(v.targets) - 4:len(v.targets)
        else:
            # nodes = graph_step.getNodes()[step] #graph_step.getNodes()[step - 4:step]
            # pos = []
            for i in range(len(nodes)):
                for (k, v) in nodes.items():
                    if k in common_nodes and len(v.targets):
                        # print(k , v.targets)
                        # if len(v.targets) == len(pos[i]):
                        pos[i] = torch.Tensor(v.targets)

        for i in range(pos.size()[0]):
            if len(pos[i]) < len(common_nodes):
                for j in range(batch_size): #len(common_nodes) - len(pos[i])
                    pos[i] = torch.Tensor([[0.0, 0.0]])
        pos = pos.permute(1, 0, 2)
    else:
        common_nodes, _, _ = get_common_nodes(curr_graph=graph_step, frame=step,
                                    curr_graph_nodes=curr_graph_nodes.getNodes())

        # pos = {} # TODO: fix memory assignment
        if step == 0:
            step += 1
        # when using minibatch setting step > len(common_nodes)
        if step >= len(common_nodes):
            step = len(common_nodes)-1
        if len(common_nodes) and len(common_nodes[step]):
            for i in range(step, len(graph_step)):
                nodes = []
                for k, x in enumerate(graph_step[i]):
                    nodes.append((list(x.keys())[0], k))
                for k, x in enumerate(nodes):
                    # append 12 steps
                    if k >= max_len: #or i >= len(common_nodes):
                        break
                    elif x[0] in common_nodes[step]: #in common_nodes[i]:
                        try:
                           pos[x[0]].append(graph_step[i][k][x[0]])
                        except KeyError:
                            if len(pos):
                                pos[x[0]] = [graph_step[i][k][x[0]]]
                            else:
                                pos = {x[0]: [graph_step[i][k][x[0]]]}

        return pos, common_nodes

    if isinstance(pos, list):
        pos = np.stack(pos)
        pos = torch.Tensor(pos)
        pos = pos.permute(1, 0, 2)
        if np.ndim(pos) >= 3:
            pos = pos.squeeze(2)
        else:
            pos = pos.squeeze()

    pos = pos.cuda()
    return pos, common_nodes,exc_node

def get_common_nodes(curr_graph, frame, curr_graph_nodes=None):

    if frame == 0:
        frame += 1
    exc_node = []

    if not isinstance(curr_graph, list):
        n1 = curr_graph.getNodes()  # new graph

        if frame >= len(n1):
            frame = len(n1)-1
        common_nodes = set(n1[frame]) & set(n1[frame - 1])#, reverse=False #sorted()
        l1 = set(n1[frame]) - set(n1[frame - 1])
        exc_node = sorted(l1 if len(l1) else set(n1[frame - 1]) - set(n1[frame]), reverse=False)
        return common_nodes, n1, exc_node
    else: # for targets
        n1 = [] # TODO: fix memory assignment

        common_nodes = set(curr_graph_nodes[frame])
        common_list = []
        for i in range(len(curr_graph)):
                n1.append(np.zeros(shape=(len(curr_graph[i])), dtype=np.float32))
                for k, x in enumerate(curr_graph[i]):
                    n1[i][k] = list(x.keys())[0]

                inter = set(n1[i]) & set(common_nodes)

                if len(inter):
                    common_list.append(sorted(inter))

    common_list = list(filter(None, common_list))
    return common_list, n1, exc_node


def update_hidden_state_tensor(prev_hidden, prev_cell, cut_idx, max_len=40):
    old_len = prev_hidden.size()[1]
    if max_len < old_len:
        max_len = old_len
    cat_len = abs(len(cut_idx) - max_len)

    if cat_len or old_len != max_len:
        prev_hidden = torch.index_select(prev_hidden.cpu(), index=torch.LongTensor(cut_idx[0:max_len]), dim=1).cuda()#
        prev_cell = torch.index_select(prev_cell.cpu(), index=torch.LongTensor(cut_idx[0:max_len]), dim=1).cuda()#
        prev_cell = torch.cat((prev_cell, torch.zeros(batch_size, cat_len, prev_hidden.size()[2]).cuda()), dim=1) #
        prev_hidden = torch.cat((prev_hidden, torch.zeros(batch_size, cat_len, prev_hidden.size()[2]).cuda()), dim=1) #
        # torch.cuda.empty_cache()

    return prev_hidden, prev_cell, cat_len
