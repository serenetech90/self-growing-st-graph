
# Title     : TODO: self-growing v4, accurate indexing + adjecancy matrix calculation
# Objective : TODO: correct indexing of pedestrian hidden states and outputs in fixed tabulated format
#                  + self-growing criterion

import torch
import torch.cuda
import argparse
import time
import gc
from read_data import *
import online_graph as Graph
import lstm_network as lstmS
import criterion as cr
import torch.optim.lr_scheduler as scheduler
import torch.nn

def main():
    train_args = argparse.ArgumentParser()

    train_args.add_argument('--batch_size' ,  type=int, default=8)
    train_args.add_argument('--seq_length', type=int, default=10) # 12 for sdd, 10 for crowds
    train_args.add_argument('--max_len', type=int, default=40) # max:40 for ucy-univ, others are 20
    # train_args.add_argument('--lstm_size', type=int, default=256)
    train_args.add_argument('--pred_len', type=int, default=12)
    train_args.add_argument('--num_layers', type=int, default=4)
    # train_args.add_argument('--num_epochs', type=int, default=300)
    train_args.add_argument('--learning_rate', type=float, default=0.00001, help='')
    # train_args.add_argument('--leaveDataset', type=float, default=0, help='')
    train_args.add_argument('--decay_rate', type=float, default=0.0001)
    # set decay rate low at the beginning then later discover its right tuning in online learning

    train_args.add_argument('--dropout', type=float, default=0.0001)
    # no need for dropout in online learning as the procedure might suffer from instabilities due to small batch size and immeditate updates after each iteration

    train_args.add_argument('--embedding_size', type=int, default=64)
    train_args.add_argument('--lambda_param', type=float, default=0.0005)
    # We'll figure out later whether high regularization (like 0.005) is useful for the convergence of GD in online setting
    # try new clip value 50
    train_args.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
    # Input and output size
    train_args.add_argument('--human_node_input_size', type=int, default=2,
                        help='Dimension of the node features')
    # train_args.add_argument('--human_input_size', type=int, default=2,
    #                     help='Dimension of the edge features')
    train_args.add_argument('--human_node_output_size', type=int, default=2,
                        help='Dimension of the node output')

    train_args.add_argument('--human_node_embedding_size', type=int, default=128, #(2*64) old = 64
                        help='Embedding size of node features')
    # train_args.add_argument('--human_human_edge_embedding_size', type=int, default=64,
    #                     help='Embedding size of edge features')

    train_args.add_argument('--human_node_rnn_size', type=int, default=256,
                        help='Size of Human Node RNN hidden state')
    train_args.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')
    train_args.add_argument('--output_size', type=int, default=2,
                            help='output layer vector size')
    # train_args.add_argument('--neighborhood_size', type=int, default=20) # 20 meter for social distance
    # # if to set interpersonal distance it shall be much smaller as it is the distance used to avoid collisions
    # # start with large social neighborhood

    train_args = train_args.parse_args()
    train(train_args)

# def plot(gt=None , pred=None, hold=False):
#
#     if gt is not None and pred is not None:
#         gt = gt.cpu().numpy()
#         pred = pred.data.cpu().numpy()
#
#         if not hold:
#             py.figure(figsize=[16,13])
#         for i in range(len(gt)):
#             py.plot(gt[i,:,0] , gt[i,:,1], 'g*-')
#             py.plot(pred[i,:,0], pred[i,:,1], 'r*-')
#
#         py.axis([0, 20 , 0 , 15])
#
#         py.show()
#
#     return

def train(args):
    torch.cuda.set_device(0)

    step_start = 0
    for idx in {7}:# 0, 1, 2, 3, 4, 7
        dataloader = DataLoader(args=args, sel=idx ,start=9, datasets=range(10))

        more_steps = True
        log_param = str(dataloader.current_dir).replace('data', 'log')
        log_param = open(os.path.join(log_param, 'param_curve_{0}_{1}_{2}_mse.txt'
                                      .format(idx, args.pred_len, args.batch_size)), 'w')

        log_directory = str(dataloader.current_dir).replace('data', 'log')
        log_file_curve = open(os.path.join(log_directory, 'log_curve_{0}_{1}_{2}_mse.txt'
                                           .format(dataloader.dataset_pointer, args.pred_len, args.batch_size)), 'w')

        error_file = open(os.path.join(log_directory, 'error_{0}_{1}_{2}_mse.txt'
                                       .format(dataloader.dataset_pointer, args.pred_len, args.batch_size)), 'w')

        # Save directory
        save_directory = str(dataloader.current_dir).replace('data',
                                                             'save')  # '/home/serene/Documents/InVehicleCamera/save_kitti/'

        checkpoint_path = os.path.join(save_directory, 'srnn_model_{0}_{1}_{2}_batch_{3}_start_{4}_mse.tar')

        # Initialize net
        net = lstmS.lstm_network(args, file_hook=log_param).cuda()

        optimizer = torch.optim.Adagrad(net.parameters(),lr=  args.learning_rate,
                                        lr_decay=args.lambda_param,
                                        weight_decay=args.decay_rate)
        # TODO set annealing scheduler
        #  try adagrad for OGD
        schedul = scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=args.decay_rate,
                                              patience=1)

        step_count = 0
        # args.diff = args.seq_length #dataloader.diff
        args.seed = dataloader.seed

        graph = Graph.online_graph(args)
        dataloader.reset_data_pointer()
        differ = False
        clip_gradient = torch.nn.utils.clip_grad_norm_
        # targets = []
        reset_every = 5

        new_hidden = None
        new_cell = None
        del_graph = False
        loss = 0


        norm_ade_final = []
        norm_fde_final = []
        no_targ = 0
        while more_steps:

            norm_ade = 0
            norm_fde = 0

            retain = step_count % reset_every
            plt_targets = []
            plt_pred = []
            # torch.cuda.empty_cache()
            # torch.cuda.reset_max_memory_cached(device=0)

            # print('============================')
            # print("Memory used: {:.2f} M".format(torch.cuda.memory_allocated() / 1024 / 1024))
            # print('============================')

            if retain == 0 and step_count != 0:
                graph.reset_graph(step_count)
                # del prev_cell, prev_hidden
                gc.collect()
                # del net
                clip_gradient(net.parameters(), args.grad_clip)
                for p in list(net.parameters()):
                    if not p.grad is None:
                        p.data.add_(-args.learning_rate, p.grad.data)
                prev_hidden = torch.div(prev_hidden, args.grad_clip)
            start = time.time()

            batch, targets, frame = dataloader.next_step() #randomUpdate=False no shuffling since we are following online process

            if len(batch) == 0 or len(batch[frame - args.seq_length]) == 0:
                more_steps = False
                continue

            graph_step = graph.ConstructGraph(current_batch=batch, framenum=step_count)
            if step_count == 0:
                prev_hidden = torch.zeros(args.batch_size, args.max_len, args.human_node_rnn_size).cuda()
                prev_cell = torch.zeros(args.batch_size, args.max_len, args.human_node_rnn_size).cuda()

            if step_count > 0:
                mem1 = torch.cuda.memory_allocated()
                pseudo_count = step_count
                if step_count >= len(graph_step.getNodes()):
                    pseudo_count = len(graph_step.getNodes()) - 1
                    graph_step.step = pseudo_count
                    graph.linkGraph(curr_graph=graph_step, frame=pseudo_count)
                else:
                    graph.linkGraph(curr_graph=graph_step, frame=step_count)

                if step_count % args.batch_size == 0 and step_count >= args.batch_size:
                    del_graph = True
                    cr.get_node_positions(graph_step=targets,
                                          curr_graph_nodes=graph_step,
                                          step=pseudo_count, istargets=True)
                    if new_hidden is not None:
                        prev_hidden = prev_hidden.add(new_hidden)
                        prev_cell = prev_cell.add(new_cell)

                    traj = graph_step.getNodes()[pseudo_count]
                    velocities = torch.zeros((args.max_len, 1)).cuda()

                    for k, (i, v) in zip(range(len(traj)), traj.items()):
                        begin_traj = v.getTargets()[pseudo_count:pseudo_count + args.batch_size]
                        if len(begin_traj):
                            vel = \
                                np.linalg.norm(np.subtract(begin_traj[len(begin_traj) - 1], begin_traj[0]), ord=2) / (
                                        (len(begin_traj) * 10) / 30)
                            graph_step.nodes[pseudo_count][i].vel = vel
                            velocities[k] = vel

                    pred_traj, new_hidden, new_cell, new_len, exc_node, disp_len,_ = net(graph_step=graph_step,
                                            velocity=velocities, prev_hidden=prev_hidden,
                                            prev_cell=prev_cell, differ=differ, retain=retain)

                    if (torch.sum(torch.isnan(pred_traj)) > 0) \
                            or (torch.sum(torch.isinf(pred_traj)) > 0):
                        print("nan or inf predictions")
                else:
                    step_count += 1
                    continue

                # Check variation in size before passing new hidden states to update older matrix
                old_len = prev_hidden.size()[1]
                expand = old_len < new_hidden.size()[1]  # old_len; set max_size to 10 pedestrians per Frame

                cut_idx = list(set(graph_step.getNodes()[pseudo_count - 1]) - set(exc_node))
                cut_idx -= np.ones((len(cut_idx)))

                if expand:
                    extra_size = torch.zeros(args.batch_size, new_len - old_len, args.human_node_rnn_size).cuda()
                    prev_hidden = torch.cat((extra_size, prev_hidden), dim=1).cuda()
                    prev_cell = torch.cat((extra_size, prev_cell), dim=1).cuda()

                curr_nodes = graph_step.getNodes()[step_count]

                out_targets, common_nodes = cr.get_node_positions(graph_step=targets,
                                                                  nodes=curr_nodes,
                                                                  pos={}, #torch.zeros((len(curr_nodes), args.batch_size, 2)),
                                                                  max_len=new_len,
                                                                  curr_graph_nodes = graph_step,
                                                                  step=pseudo_count, istargets=True)

                if len(targets):
                    # TODO: make adaptive prediction length according to true trajectories (DONE)
                    for i,v in out_targets.items():
                        if pseudo_count >= len(v):
                            idx_start = pseudo_count-len(v) <=len(v) if pseudo_count-len(v) else 0
                            targets_at = v[idx_start:idx_start+args.pred_len]
                        else:
                            targets_at = v[pseudo_count-args.batch_size+1:int(pseudo_count+args.pred_len/2)+1]

                        if not len(targets_at):
                            continue

                        targets_at = torch.Tensor(np.stack(targets_at)).cuda()
                        # out_sh = pred_traj.size()
                        # rang = len(targets_at)
                        pred_traj_at = graph_step.nodes[pseudo_count][i].getPrediction()
                        if len(pred_traj_at):

                            out_sh = pred_traj_at.size()
                            if len(targets_at):
                                no_targ += 1
                                err, norm_1, norm_2 = cr.online_BP(outputs=pred_traj_at,
                                                                target=targets_at)

                                # loss_cpu = loss_cpu + (err/out_sh[0])
                                loss += (err/out_sh[0])
                                norm_ade += torch.sum(norm_1) #/12 #.append(torch.sqrt(torch.sum(norm_1, 1).sum(0)))
                                norm_fde += torch.sum(norm_2) #.append(torch.sqrt(torch.sum(norm_2)))

                                plt_targets.append(targets_at)
                                plt_pred.append(pred_traj_at)

                    norm_ade_final.append(norm_ade)#/(no_targ))
                    norm_fde_final.append(norm_fde)#/no_targ)

                    end = time.time()
                    # for ind in range(len(norm_ade)):
                    #     print('ADE = ',float(torch.sqrt(torch.sum(norm_ade[ind].data)) /(no_targ*12)))
                    #     print('FDE = ', float(torch.sqrt(torch.sum(norm_fde[ind]))/no_targ))

                    if no_targ:
                        back_start = time.time()
                        loss.backward(retain_graph=True) # allocate more memory
                        # need retain_graph=True in te variational setting cuz
                        # after the first backward proc variables are freed to save memory, hence in variational method
                        # the variables and computational graph needs to be retained for repeating the backward proc mutliple times
                        back_end = time.time()
                        print("backward pass took {0}".format((back_end - back_start)))

                    print('memory consumption by this iteration = ',
                          (torch.cuda.memory_allocated() - mem1) / 1024 / 1024)

                    # update weights
                    optimizer.step() #closure=cr.online_BP
                    schedul.step(loss)
                    optimizer.zero_grad()

                    print('lr= ', net.args.learning_rate)

                    log_param.write('\nprediction trajectory\n' + str(pred_traj) + '\ntargets\n' + str(out_targets) + '\n\n**************\n')

                print("took {0} seconds".format((end - start)))
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated(0)
                torch.cuda.reset_max_memory_cached(0)

            print('step: ', step_count, '  frame: ', frame)
            # for ind in range(len(norm_ade)):
            #     log_file_curve.write(str(step_count) + ',' + str(float(loss.data.cpu()))
            #                          + ',' + str(float(norm_ade[ind].data)) +
            #                          ',' + str(float(norm_fde[ind].data)) + ',' + str(end - start) + ',' + str(
            #         back_end - back_start)  + '\n')

            if del_graph:
                del graph_step, targets, targets_at
                del out_targets, common_nodes
                del_graph = False
            step_count += 1

            # save online model parameters after every adaptation step-wise (online) (opposite:batch-wise (offline))
            print('Saving new adaptation')
            torch.save({
                'epoch': step_count,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path.format(step_count, idx, args.pred_len, args.batch_size, step_start))

            if step_count >= 80:
                break

        norm_ade_final = torch.stack(norm_ade_final)
        norm_fde_final = torch.stack(norm_fde_final)

        log_file_curve.write(str(no_targ))
        print('MSE Loss = ' + str(torch.sqrt(torch.sum(loss)) / no_targ) +
                         'ADE = ' + str(np.average(torch.sqrt(norm_ade_final).cpu().data/(no_targ*args.pred_len))) #np.sum(list(norm_ade.cpu().data))
                         + 'FDE = ' + str(np.average(torch.sqrt(norm_fde_final).cpu().data/no_targ))) #np.sum(list(norm_fde.cpu().data))

        # error_file.write('MSE Loss = ' + str(torch.sqrt(torch.sum(loss))/no_targ) +
        #                  'ADE = '+  str(torch.sqrt(torch.sum(norm_ade) )/(no_targ * args.pred_len))
        #                  + 'FDE = ' + str(torch.sqrt(torch.sum(norm_fde) )/no_targ ))

        log_file_curve.close()
        log_param.close()
        error_file.close()

if __name__ == '__main__':
    main()

