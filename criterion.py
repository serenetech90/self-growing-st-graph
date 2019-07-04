''' Function that implements loss for online prediction
squared loss ?? tends to maximize error due to using squares .. can this be beneficial for double penalizing
the estimation error at current steps therefore lowering errors exponentially at a constant rate
so easier training ?
'''

import numpy as np
import torch
import torch.nn as nn


def online_BP(outputs, target):
    out_sh = outputs.size()

    # outputs = torch.reshape(outputs, shape=(out_sh[0], int(out_sh[1]/2) ,2))
    out_sh = outputs.size()
    end = out_sh[1] - 1  # for 1x12 sequence
    end_tg = target.size()[0] - 1  # for nx12 sequence
    end_pred = outputs.size()[0] - 1
    # inf: make lr smaller/ divide by zero
    # nan: make lr smaller/ sqrt(negative value)
    # watch over final layer outputs if they explode in value
    # permute target sequences to be same order of axes as predicted sequences [nx12x2]
    # target = target.permute(1,0,2)
    # define loss function
    loss = nn.MSELoss()  # size_average is by default true , n = number of present nodes in current frame
    err = loss(outputs[0:len(target)], target)  # divide by (1d*2d*3d) not by batch_size

    norm_ade = (outputs[0:len(target)] - target).pow(2)
    # norm_ade = (torch.norm((outputs[0:len(target)] - target), p=2))  # / out_sh[0]
    # norm_ade = torch.sqrt(torch.pow(torch.sum(outputs - target), 2.0)) #/ (out_sh[0] * out_sh[1])
    if out_sh[0] > 1:
        norm_fde = outputs[end] - target[end_tg]
        norm_fde = norm_fde.pow(2)
        # norm_fde = torch.norm((outputs[end_pred] - target[end_tg]), p=2.0)
        # torch.sqrt(torch.pow(torch.sum(outputs[end_tg] - target[end_tg])
        # , 2.0)) # / (out_sh[0])
    else:
        norm_fde = (outputs[:, end_tg] - target[:, end_tg]).pow(2)
        # norm_fde = torch.norm((outputs[:, end_pred] - target[:, end]), p=2.0)
        # torch.sqrt(torch.pow(torch.sum(outputs[:, end] - target[:,end])
        #                                 , 2.0))
        # print("predicted trajectory" , outputs)
        # print("\n \n ground-truth ", targets)
    # TODO: check the RMSE and see how square rooting error affect the learning convergence
    # err = torch.sqrt(loss(outputs, targets))

    return err, norm_ade, norm_fde


def get_node_positions(graph_step, step=0, max_len=20, curr_graph_nodes=None, istargets=False):
    exc_node = []
    if not istargets:
        common_nodes, _, exc_node = get_common_nodes(curr_graph=graph_step, frame=step)
          # sorted()
        # doesnt work with some ped_id that are not organized by pattern sorted()

        pos = []  # np.empty(shape=(dim0, 2), dtype=np.float32)
        if len(common_nodes) == 0:
            nodes = graph_step.getNodes()[step]
            pos = torch.zeros(( len(nodes), 4, 2))
            for i in range(len(nodes)):
                # pos.append([])
                for (k, v) in nodes.items():
                    if len(v.targets) > step:
                        pos[i] = torch.Tensor(v.targets[step - 4:step])
                    else:
                        pos[i] = torch.Tensor(v.targets[len(v.targets) - 4:len(v.targets)])

            pos = pos.permute(1, 0, 2)
        else:
            nodes = graph_step.getNodes()[step - 4:step]
            pos = []
            for i in range(len(nodes)):
                pos.append([])
                for (k, v) in nodes[i].items():
                    if k in common_nodes:
                        pos[i].append(v.targets)

        for i in range(len(pos)):
            if len(pos[i]) < len(common_nodes):
                for j in range((len(common_nodes) - len(pos[i]))):
                    pos[i].append([[0, 0]])
                    # rpt = np.repeat([0, 0], (len(common_nodes) - len(pos[i])), axis=0)
                    # pos[i].append(rpt.tolist())

    else:
        common_nodes, _, _ = get_common_nodes(curr_graph=graph_step, frame=step,
                                    curr_graph_nodes=curr_graph_nodes.getNodes())

        # pos = np.empty(shape=(len(common_nodes), len(graph_step[0]),2), dtype=np.float32)
        pos = {} # TODO: fix memory assignement

        if step == 0:
            step += 1
        # when using minibatch setting step > len(common_nodes)
        if step >= len(common_nodes):
            step = len(common_nodes)-1

        if len(common_nodes) and len(common_nodes[step]):
            for i in range(step , len(graph_step)):
                # idx = 0
                # if len(graph_step[i]) > max_len:
                #     max_len = len(graph_step[i])
                # pos.append(np.zeros(shape=(1,max_len ,2))) # 10 max number of pedestrians pre frame
                nodes = []
                for k, x in enumerate(graph_step[i]):
                    nodes.append((list(x.keys())[0] , k))
                for k,x in enumerate(nodes):
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
                                pos = {x[0]: [ graph_step[i][k][x[0]] ]}

        return pos, common_nodes

    if isinstance(pos, list):
        pos = np.stack(pos)
        if np.ndim(pos) > 3:
            pos = pos.squeeze(2)
            # pos = torch.Tensor(pos).permute(1, 0, 2)
        else:
            pos = pos.squeeze()

        pos = torch.Tensor(pos)

    return pos , common_nodes,exc_node


def get_common_nodes(curr_graph, frame, curr_graph_nodes=None):
    if frame == 0:
        frame += 1
    exc_node = []

    if not isinstance(curr_graph, list):
        n1 = curr_graph.getNodes()  # new graph

        if frame >= len(n1):
            frame = len(n1) - 1
        common_nodes = set(n1[frame]) & set(n1[frame - 1])  # , reverse=False #sorted()
        l1 = set(n1[frame]) - set(n1[frame - 1])
        exc_node = sorted(l1 if len(l1) else set(n1[frame - 1]) - set(n1[frame]), reverse=False)
        return common_nodes, n1, exc_node
    else:  # for targets
        n1 = []  # TODO: fix memory assignment

        common_nodes = set(curr_graph_nodes[frame])
        common_list = []
        for i in range(len(curr_graph)):
            n1.append(np.zeros(shape=(len(curr_graph[i])), dtype=np.float32))
            for k, x in enumerate(curr_graph[i]):
                n1[i][k] = list(x.keys())[0]

            inter = set(n1[i]) & set(common_nodes)

            if len(inter):
                common_list.append(sorted(inter))

    # l1 = set(n1[frame]) - set(n1[frame - 1])
    # exc_node = l1 if len(l1) else set(n1[frame-1]) - set(n1[frame])# sorted, reverse=False)
    # common_list = list(filter(None, common_list))
    return common_list, n1, exc_node


def update_hidden_state_tensor(prev_hidden, prev_cell, cut_idx, max_len=20):
    old_len = prev_hidden.size()[1]
    if max_len < old_len:
        max_len = old_len
    cat_len = len(cut_idx) - max_len

    # if cat_len > 0:
    #     prev_hidden = torch.cat((prev_hidden, torch.zeros(1, cat_len, prev_hidden.size()[2])), dim=1)
    #     prev_cell = torch.cat((prev_cell, torch.zeros(1, cat_len, prev_hidden.size()[2])), dim=1)

    if old_len < max_len:
        end = old_len
    else:
        end = max_len

    if (len(cut_idx) < max_len):
        # if prev_hidden.size()[1] < max_len
        prev_hidden = torch.index_select(prev_hidden, index=torch.LongTensor(cut_idx[0:end]), dim=1)
        # prev_cell = torch.index_select(prev_cell, index=torch.LongTensor(cut_idx[0:max_len]), dim=1)
        prev_hidden = torch.cat((prev_hidden, torch.zeros(len(prev_hidden), abs(cat_len), prev_hidden.size()[2])), dim=1)

        # prev_cell = torch.cat((prev_cell, torch.zeros(1, abs(cat_len), prev_hidden.size()[2])), dim=1)
        # elif cat_len < 0:
        #     cat_len *= -1
        #     prev_hidden = torch.cat((prev_hidden, torch.zeros(1, cat_len, prev_hidden.size()[2])), dim=1)
        #     prev_cell = torch.cat((prev_cell, torch.zeros(1, cat_len, prev_hidden.size()[2])), dim=1)

    return prev_hidden, prev_cell, cat_len
