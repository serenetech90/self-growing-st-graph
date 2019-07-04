# Title     : TODO: self-growing v4, accurate indexing + adjecancy matrix calculation
# Objective : TODO: correct indexing of pedestrian hidden states and outputs in fixed tabulated format
#                  + self-growing criterion
# Created by: serene
# Created on: 10/04/19


import time
import os
import args_parser as parser
import torch
import torch.nn
import torch.optim.lr_scheduler as scheduler
from torch.autograd import Variable

import criterion as cr
import lstm_network as lstmS
import online_graph as Graph
from read_data import *


def main():
    train_args = parser.arg_parse()
    train(train_args)


def train(args):
    # for i in range(6):
    # online training
    # for j in {6}:
    for i in {4}: #, 5 , 6
            dataloader = DataLoader(args=args, sel=i, start=6, datasets=range(7))

            more_steps = True
            log_param = str(dataloader.current_dir).replace('data', 'log')
            log_param = open(os.path.join(log_param, 'param_curve_{0}_{1}_{2}_mse.txt'
                                          .format(dataloader.dataset_pointer, args.pred_len, args.batch_size)), 'w')

            log_directory = str(dataloader.current_dir).replace('data', 'log')
            log_file_curve = open(os.path.join(log_directory, 'log_curve_{0}_{1}_{2}_mse.txt'
                                               .format(dataloader.dataset_pointer, args.pred_len, args.batch_size)), 'w')

            error_file = open(os.path.join(log_directory, 'error_{0}_{1}_{2}_mse.txt'
                                           .format(dataloader.dataset_pointer, args.pred_len, args.batch_size)), 'w')

            # Save directory
            save_directory = str(dataloader.current_dir).replace('data',
                                                                 'save')  # '/home/serene/Documents/InVehicleCamera/save_kitti/'
            # save_directory += str(args.leaveDataset) + '/'

            checkpoint_path = os.path.join(save_directory, 'srnn_model_{0}_{1}_{2}_batch_{3}_mse.tar')

            # Initialize net
            net = lstmS.lstm_network(args, file_hook=log_param)
            # net.cuda()
            # net.humanNode._parameters
            optimizer = torch.optim.Adagrad(net.parameters(), lr=args.learning_rate,
                                            lr_decay=args.lambda_param,
                                            weight_decay=args.decay_rate)
            # TODO set annealing scheduler
            # later try adagrad for OGD
            schedul = scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=args.decay_rate,
                                                  patience=1)

            step_count = 0
            # args.diff = args.seq_length #dataloader.diff
            args.seed = dataloader.seed

            graph = Graph.online_graph(args)
            dataloader.reset_data_pointer()
            differ = False
            clip_gradient = torch.nn.utils.clip_grad_norm_
            targets = []
            reset_every = 100

            new_hidden = None
            new_cell = None

            loss = 0
            norm_ade = []
            norm_fde = []
            no_targ = 0

            while more_steps:
                retain = True
                if step_count % reset_every == 0 and step_count != 0:
                    graph.reset_graph(step_count)
                    # del net
                    clip_gradient(net.parameters(), args.grad_clip)
                    for p in list(net.parameters()):
                        if not p.grad is None:
                            p.data.add_(-args.learning_rate, p.grad.data)

                    prev_hidden = torch.div(prev_hidden, args.grad_clip)
                    # net = lstmS.lstm_network(args, file_hook=log_param)

                if step_count > 350:
                    break
                start = time.time()
                # start = time.clock()
                # velocities = []

                batch, targets, frame = dataloader.next_step(targets)
                # randomUpdate=False no shuffling since we are learning from sequential data
                # try shuffling input with multistack lstm === NOT working / worse results

                if len(batch) == 0 or len(batch[frame-args.seq_length]) == 0:
                    # more_steps = False
                    continue

                graph_step = graph.ConstructGraph(current_batch=batch, framenum=step_count)
                if step_count == 0:
                    prev_hidden = Variable(torch.zeros(args.num_layers,args.max_len, args.human_node_rnn_size))
                    prev_cell = Variable(torch.zeros(args.num_layers,args.max_len, args.human_node_rnn_size))

                if step_count > 0:
                    pseudo_count = step_count
                    if step_count >= len(graph_step.getNodes()):
                        pseudo_count = len(graph_step.getNodes()) - 1
                        graph_step.step = pseudo_count
                        graph.linkGraph(curr_graph=graph_step, frame=pseudo_count)
                    else:
                        graph.linkGraph(curr_graph=graph_step, frame=step_count)

                    if step_count % args.batch_size == 0 and step_count >= args.batch_size:
                        cr.get_node_positions(graph_step=targets,
                                              curr_graph_nodes=graph_step,
                                              step=pseudo_count, istargets=True)
                        if new_hidden is not None:
                            # prev_hidden = Variable(torch.zeros_like(new_hidden))
                            # prev_cell = Variable(torch.zeros_like(new_cell))
                            prev_hidden += new_hidden
                            prev_cell += new_cell

                        traj = graph_step.getNodes()[pseudo_count]
                        velocities = torch.zeros((args.max_len, 1))

                        for k ,(i, v) in zip(range(len(traj)) ,traj.items()):
                            begin_traj = v.getTargets()[pseudo_count:pseudo_count + args.batch_size]
                            if len(begin_traj):
                                vel = \
                                    np.linalg.norm(np.subtract(begin_traj[len(begin_traj) - 1], begin_traj[0]), ord=2) / (
                                                (len(begin_traj) * 10) / 30)
                                graph_step.nodes[pseudo_count][i].vel = vel
                                velocities[k] = vel

                        pred_traj, new_hidden, new_cell, new_len, \
                        exc_node, disp_len = net(graph_step=graph_step, velocity= velocities ,prev_hidden=prev_hidden,
                                                 prev_cell=prev_cell, differ=differ)

                        if (torch.sum(torch.isnan(pred_traj)) > 0) \
                                or (torch.sum(torch.isinf(pred_traj)) > 0):
                            print("nan or inf predictions")

                        # TODO check the right way to maintain previous learned information along
                        # time without catastrophic forgetting
                        # currently, we do propagation
                        # TODO what about accumulating new states with prev_states using
                        # TODO addition? multiplication?

                        out_targets, common_nodes = cr.get_node_positions(graph_step=targets, max_len=new_len,
                                                                          curr_graph_nodes=graph_step,
                                                                          step=pseudo_count, istargets=True)

                    else:
                        step_count += 1
                        continue

                    # Check variation in size before passing new hidden states to update older matrix
                    old_len = prev_hidden.size()[1]
                    expand = old_len < new_hidden.size()[1]  # old_len; set max_size to 10 pedestrians per Frame

                    cut_idx = list(set(graph_step.getNodes()[pseudo_count - 1]) - set(exc_node))
                    cut_idx -= np.ones((len(cut_idx)))

                    if expand:
                        extra_size = torch.zeros(args.num_layers, new_len - old_len, args.human_node_rnn_size)
                        prev_hidden = torch.cat((extra_size, prev_hidden), dim=1)
                        prev_cell = torch.cat((extra_size, prev_cell), dim=1)

                    # targets = targets.permute(1, 0, 2)

                    # TODO: need to code the online BP method (RTRL for computing derivatives and update weights\
                    # TODO  after processing each step of the trajectory == online update)
                    # calculate derivative of hidden vectors representing motion features (h*)
                    # pred_traj = torch.reshape(pred_traj, shape=(out_sh[0], int(out_sh[1] / 2), 2))
                    # pred_traj = pred_traj.permute(1, 0, 2)
                    if not len(targets):
                        #     free up gradients and dont compute loss, useless predictions
                        pass
                    else:
                        # out_sh = pred_traj.size()
                        # TODO: make adaptive prediction length according to true trajectories (DONE)
                        # cut predictions down to actual ground-truth trajectories
                        # if len(out_targets) > len(pred_traj):
                        #     out_targets = out_targets.index_select(dim=0, index=torch.LongTensor(cut_idx))
                        # no need to check expand on targets and pred_traj
                        # elif expand:
                        #     targets = torch.cat((extra_size, targets), dim=1)
                        for i, v in out_targets.items():
                            if pseudo_count >= len(v):
                                idx_start = pseudo_count - len(v) <= len(v) if pseudo_count - len(v) else 0
                                targets_at = v[idx_start:idx_start + args.pred_len]
                            else:
                                targets_at = v[pseudo_count:pseudo_count + args.pred_len]

                            if not len(targets_at):
                                continue

                            targets_at = torch.Tensor(np.stack(targets_at))
                            out_sh = pred_traj.size()

                            rang = len(targets_at[0])
                            pred_traj_at = graph_step.nodes[pseudo_count][i].getPrediction()

                            if len(pred_traj_at):
                                # pred_traj_at = torch.reshape(pred_traj_at, shape=(int(out_sh[0] / args.pred_len), 2))[0: rang]
                                # rang = round(float(torch.sum(targets[i] != 0) / 2)) # >0 doesnt count negative coordinates
                                # # pred_traj_at = pred_traj[i][0: rang]
                                # targets_at = targets[i][0: rang]
                                # pred_traj_at = pred_traj[i][0: rang]
                                out_sh = pred_traj_at.size()
                                if len(targets_at):
                                    no_targ += 1
                                    err, norm_1, norm_2 = cr.online_BP(outputs=pred_traj_at,
                                                                       target=targets_at)

                                    loss += (err / out_sh[0])
                                    norm_ade.append(torch.sum(norm_1, 1).sum(0))
                                    norm_fde.append(torch.sum(norm_2))

                        end = time.time()
                        if no_targ:
                            # norm_ade /= no_targ  # len(targets)
                            # norm_fde /= no_targ  # len(targets)
                            # loss /= len(out_targets)

                            back_start = time.time()
                            # back_start = time.clock()
                            loss.backward(retain_graph=True)

                            # compute gradients
                            # TODO: need to code the online strategy for this gradient
                            # need retain_graph=True in te variational setting cuz
                            # after the first backward proc variables are freed to save memory, hence in variational method
                            # the variables and computational graph needs to be retained for repeating the backward proc mutliple times
                            back_end = time.time()
                            print("Loss = ", loss.data, ' norm_ade, norm_fde = ', norm_ade, norm_fde)


                        # back_end = time.clock()
                        if math.isnan(loss) or math.isinf(loss):
                            clip_gradient(net.parameters(), args.grad_clip)
                            for p in list(net.parameters()):
                                if not p.grad is None:
                                    p.data.add_(-args.learning_rate, p.grad.data)
                        # print("outputs = ",  pred_traj)
                        # print("\n real_traj = ", targets)
                        print("backward pass took {0}".format((back_end - back_start)))

                        # update weights
                        optimizer.step()  # closure=cr.online_BP
                        schedul.step(loss)

                        optimizer.zero_grad()

                        # net.args.learning_rate = net.args.learning_rate * net.args.decay_rate
                        # optimizer.state_dict()['param_groups'][0]['lr'] = net.args.learning_rate

                        print('lr= ', net.args.learning_rate)
                        # TODO: calculate online estimation errors, the error shall be reducing as more steps are
                        # TODO: observed, at each observed step update the estimation for the next 12 steps
                        # graph.reset_graph()
                        # log outputs and loss

                        # log_param.write(str(net.humanNode.who) + '\n who2 \n' + str(net.humanNode.who2) + '\n who3 \n'
                        #                 + str(net.humanNode.who3))
                        log_param.write('\nprediction trajectory\n' + str(pred_traj) + '\ntargets\n' + str(
                            out_targets) + '\n\n**************\n')

                    print("took {0} seconds".format((end - start)))
                print('step: ', step_count, '  frame: ', frame)
                for ind in range(len(norm_ade)):
                    log_file_curve.write(str(step_count) + ',' + str(float(loss.data.cpu()))
                                         + ',' + str(float(norm_ade[ind].data)) +
                                         ',' + str(float(norm_fde[ind].data)) + ',' + str(end - start) + ',' + str(
                        back_end - back_start) + '\n')

                step_count += 1

                # graph_step_prev = graph_step
                # save online model parameters after every adaptation step-wise (online) (opposite:batch-wise (offline))
                print('Saving new adaptation')
                torch.save({
                    'epoch': step_count,
                    'state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path.format(step_count, dataloader.dataset_pointer, args.pred_len, args.batch_size))


            norm_ade = torch.stack(norm_ade)
            norm_fde = torch.stack(norm_fde)

            # norm_ade = torch.sqrt(torch.sum(norm_ade, 2)).sum(1)
            # norm_fde = torch.sqrt(torch.sum(norm_fde, 2)).sum(1)
            log_file_curve.write(str(no_targ))
            error_file.write('MSE Loss = ' + str(torch.sqrt(torch.sum(loss)) / no_targ) +
                             ' rmse ADE = ' + str(torch.sqrt(torch.sum(norm_ade)) / (no_targ * args.pred_len))
                             + 'rmse FDE = ' + str(torch.sqrt(torch.sum(norm_fde)) / no_targ))

            log_file_curve.close()
            log_param.close()
            error_file.close()


if __name__ == '__main__':
    main()