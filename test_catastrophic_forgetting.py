import args_parser as parser
import numpy as np
import  ast
import torch
import torch.optim.lr_scheduler as scheduler
from torch.autograd import Variable
import criterion as cr
import lstm_network as lstmS
from torch.nn import functional as F


def main():
    dir = '/home/serene/Documents/self-growing-spatial-graph/data/sdd/'
    non_zero_traj = open(dir + 'non_zero_traj_datasets.txt').read().split('*')
    zero_traj = open(dir + 'zero_traj_datasets.txt').read().split(sep='*')
    non_zero_sample = []
    zero_sample = []
    synth_zero_sample = []
    max_width = 2000 # pixel
    max_height = 2000 # pixel
    numNodes = 1000
    node_edges = torch.zeros((numNodes,20))

    for i in range(numNodes):
        rand_x = np.random.sample() * max_width
        rand_y = np.random.sample() * max_height
        tmp = []
        for i in range(20):
            tmp.append([rand_x,rand_y])
        synth_zero_sample.append(tmp)

    synth_zero_sample = np.stack(synth_zero_sample)

    for str in non_zero_traj:
        try:
            non_zero_sample.append(ast.literal_eval(str))
        except SyntaxError or ValueError:
            continue
    non_zero_sample = np.stack(non_zero_sample)

    for str in zero_traj:
        try:
            zero_sample.append(ast.literal_eval(str))
        except SyntaxError or ValueError:
            continue
    zero_sample = np.stack(zero_sample)
    zero_sample = torch.Tensor(zero_sample)

    # random selection of trajectory set
    # for i in range(1000):
    #    ch = np.random.random_integers(low=0 , high=1)
    selection_arr = np.random.geometric(p=0.50, size=numNodes)

    balanced_sample = np.zeros((numNodes,20,2)) # 70 samples & 70
    a = (synth_zero_sample[selection_arr != 1])
    balanced_sample[0:len(a)] = a
    b = non_zero_sample[0:numNodes][selection_arr == 1]
    balanced_sample[-len(b)-1:-1] = b
    balanced_sample = np.stack(balanced_sample)

    np.random.shuffle(balanced_sample)

    balanced_sample = torch.Tensor(balanced_sample)

    args = parser.arg_parse()

    log_param = open('/home/serene/Documents/self-growing-spatial-graph/test/balanced_param_curve.txt', 'w')
    # log_param = open(os.path.join(log_param, 'param_curve_{0}_{1}.txt'
    #                               .format(dataloader.dataset_pointer, args.pred_len)), 'w')

    net = lstmS.lstm_network(args, file_hook=log_param)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=0.0001,
                                    lr_decay=args.lambda_param,
                                    weight_decay=args.decay_rate)
    # TODO set annealing scheduler
    # later try adagrad for OGD
    schedul = scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=args.decay_rate,
                                          patience=1)

    targets = []
    loss = 0
    norm_ade = 0
    norm_fde = 0

    node_pos = Variable(balanced_sample).requires_grad_(False)
    prev_hidden = torch.zeros(args.num_layers, numNodes+1, args.human_node_rnn_size)
    prev_cell = torch.zeros(args.num_layers, numNodes+1, args.human_node_rnn_size)
    cell = lstmS.HumanNode(args= args)
    outputs, hidden_state, cell_state, max_len = cell(position=node_pos[:,0,:], \
                                     distance=node_edges[:,0:1], h_prev=prev_hidden,
                                     c_prev=prev_cell)

    outputs_acc = outputs[0:numNodes] + node_pos[:,0,:]  # make use of consequent predictions(incremental learning)

    # offline training on balanced sample containing both patterns (walking and standing)
    # for k in range(1,len(zero_sample)):
    for i in range(1,args.pred_len ):
        outputs, hidden_state, cell_state, max_len = cell(position=outputs[0:numNodes] + node_pos[:,i,:], \
                                 distance=node_edges[:,i:i+1], h_prev=prev_hidden,
                                 c_prev=prev_cell)

        outputs_acc = torch.cat((outputs[0:numNodes] + node_pos[:,i,:], outputs_acc), 1)
        prev_cell = cell_state
        prev_hidden = hidden_state


    out_sh = outputs_acc.size()
    outputs_acc = torch.reshape(outputs_acc, shape=(out_sh[0], int(out_sh[1] / 2), 2))

    for i in range(len(node_pos)):
        targets.append(node_pos[i,1:9,:])
        err, norm_1, norm_2 = cr.online_BP(outputs=outputs_acc[i],target=targets[i])
        loss += (err / args.pred_len)
        norm_ade += (norm_1 /args.pred_len)
        norm_fde += norm_2

    norm_ade /= len(targets)
    norm_fde /= len(targets)
    loss /= len(targets)

    print('Loss= ', loss, ' norm ade= ', norm_ade, ' norm_fde= ', norm_fde)
    # TODO transform hidden states from 1000 to 282

    # prev_hidden = prev_hidden.squeeze()
    # prev_hidden = F.linear(input=prev_hidden.t(),
    #                        weight=torch.Tensor(len(zero_sample)+1, numNodes + 1).uniform_(-1, 1))
    # prev_hidden = prev_hidden.t().unsqueeze(dim=0)
    #
    #
    # prev_cell = prev_cell.squeeze()
    # prev_cell = F.linear(input=prev_cell.t(),
    #                        weight=torch.Tensor(len(zero_sample)+1, numNodes + 1).uniform_(-1, 1))
    # prev_cell = prev_cell.t().unsqueeze(dim=0)

    zero_sample_edges = torch.zeros((len(zero_sample),20))
    numNodes = len(zero_sample) + 1
    # prev_hidden = torch.zeros(args.num_layers, len(zero_sample)+1 ,args.human_node_rnn_size)
    # prev_cell = torch.zeros(args.num_layers, len(zero_sample)+1 ,args.human_node_rnn_size)
    # sequentially pass all of zero_sample (standing pedestrians)
    # for k in range(len(zero_sample)):

    outputs_z, hidden_state, cell_state, max_len = cell(position=zero_sample[:, 0, :], \
                                                        distance=zero_sample_edges[:, 0:1],
                                                        h_prev=prev_hidden[:,0:numNodes,:],
                                                        c_prev=prev_cell[:,0:numNodes,:], differ=True)
    outputs_acc_z = outputs_z[0:len(zero_sample)] + zero_sample[:, 0, :]

    for i in range(1, args.pred_len):
        outputs_z, hidden_state, cell_state, max_len = cell(position=zero_sample[:,i,:], \
                          distance= zero_sample_edges[:,i:i+1], h_prev=prev_hidden[:,0:numNodes,:],
                          c_prev=prev_cell[:,0:numNodes,:])
        if i > 0:
            outputs_acc_z = torch.cat((outputs_z[0:len(zero_sample)] + \
                            zero_sample[:, i, :], outputs_acc_z), 1)
        else:
            outputs_acc_z = outputs_z[0:len(zero_sample)] + zero_sample[:, i, :]

        prev_cell = cell_state
        prev_hidden = hidden_state

    out_sh = outputs_acc_z.size()
    outputs_acc_z = torch.reshape(outputs_acc_z, shape=(out_sh[0], int(out_sh[1] / 2), 2))

    targets = []
    loss = 0
    norm_ade = 0
    norm_fde = 0

    for i in range(len(zero_sample)):
        targets.append(zero_sample[i, 1:9, :])
        err, norm_1, norm_2 = cr.online_BP(outputs=outputs_acc_z[i], target=targets[i])
        loss += (err / args.pred_len)
        norm_ade += (norm_1 / args.pred_len)
        norm_fde += norm_2

    norm_ade /= len(targets)
    norm_fde /= len(targets)
    loss /= len(targets)
    print('Loss= ', loss , ' norm ade= ', norm_ade, ' norm_fde= ', norm_fde)


    targets = []
    loss = 0
    norm_ade = 0
    norm_fde = 0

    # numNodes = 1000
    c = selection_arr != 1
    d = selection_arr == 1
    novel_data = non_zero_sample[0:numNodes][c[0:numNodes]]
    novel_data = torch.Tensor(np.concatenate((novel_data , synth_zero_sample[d][0:(numNodes - len(novel_data) - 1)])) )

    zero_sample_edges = torch.zeros((numNodes, 20))
    outputs_test, hidden_state, cell_state, max_len = cell(position=novel_data[:, 0, :], \
                                                      distance=zero_sample_edges[:, 0:1],
                                                      h_prev=prev_hidden,
                                                      c_prev=prev_cell,differ=True)

    outputs_acc_test = outputs_test[0:numNodes-1] + novel_data[:, 0, :]

    # for i in novel_data:
    for i in range(1,args.pred_len ):
        outputs_test, hidden_state, cell_state, max_len = \
                                cell(position=outputs_test[0:numNodes-1] + novel_data[:,i,:], \
                                distance=zero_sample_edges[:,i:i+1],
                                h_prev=prev_hidden,
                                c_prev=prev_cell)
        if i > 0:
            outputs_acc_test = torch.cat((outputs_test[0:numNodes-1] + \
                            novel_data[:, i, :], outputs_acc_test), 1)
        else:
            outputs_acc_test = outputs_test[0:numNodes-1] + novel_data[:, i, :]

    out_sh = outputs_acc_test.size()
    outputs_acc_test = torch.reshape(outputs_acc_test, shape=(out_sh[0], int(out_sh[1] / 2), 2))

    for i in range(len(novel_data)):
        targets.append(novel_data[i,1:9,:])
        err, norm_1, norm_2 = cr.online_BP(outputs=outputs_acc_test[i],target=targets[i])
        loss += (err / args.pred_len)
        norm_ade += (norm_1 /args.pred_len)
        norm_fde += norm_2

    norm_ade /= len(targets)
    norm_fde /= len(targets)
    loss /= len(targets)

    print('Loss= ', loss, ' norm ade= ', norm_ade, ' norm_fde= ', norm_fde)

    # for i, v in out_targets.items():
    #     targets_at = v[step_count:step_count + args.pred_len]
    #     if not len(targets_at):
    #         continue
    #
    #     targets_at = torch.Tensor(np.stack(targets_at))
    #     out_sh = pred_traj.size()
    #
    #     rang = len(targets_at)
    #     pred_traj_at = graph_step.nodes[step_count][i].getPrediction()
    #     if len(pred_traj_at):
    #         pred_traj_at = torch.reshape(pred_traj_at, shape=(int(out_sh[1] / 2), 2))[0: rang]
    #         # rang = round(float(torch.sum(targets[i] != 0) / 2)) # >0 doesnt count negative coordinates
    #         # # pred_traj_at = pred_traj[i][0: rang]
    #         # targets_at = targets[i][0: rang]
    #         # pred_traj_at = pred_traj[i][0: rang]
    #         out_sh = pred_traj_at.size()
    #         if len(targets_at):
    #             no_targ += 1
    #             err, norm_1, norm_2 = cr.online_BP(outputs=pred_traj_at,
    #                                                target=targets_at)
    #             loss += (err / (out_sh[0]))
    #             norm_ade += (norm_1 / (out_sh[0]))
    #             norm_fde += norm_2
    #             # norm_fde += (norm_2 / out_sh[0])
    # # end = time.time()
    # end = time.time()
    # if no_targ:
    #     norm_ade /= no_targ  # len(targets)
    #     norm_fde /= no_targ  # len(targets)
    #     loss /= len(out_targets)

    return 0


if __name__ == '__main__':
    main()
