import torch.cuda
from torch.autograd import Variable
import args_parser as parser
import time
from read_data import *
import online_graph as Graph
import lstm_network as lstmS
import criterion as cr
# import turtle
# import torch.optim.lr_scheduler as scheduler
import torch.nn
from matplotlib import pyplot as py
from matplotlib import patches as pat
import numpy as np

def plot(gt=None , pred=None, hold=True):
    # t = turtle.Turtle()
    if gt is not None and pred is not None:
        # gt = gt.cpu().numpy()
        # pred = pred.data.cpu().numpy()

        gt = np.stack(gt)
        pred = np.stack(pred)
        x_dir = []
        y_dir = []
        fig = py.figure()
        ax = fig.add_subplot(1,1,1)

        if not hold:
            py.figure(figsize=[1322,1079])

        for i in range(len(gt)):
            py.plot(gt[i,:,0] , gt[i,:,1], 'go-')
            # py.plot(pred[i,:,0], pred[i,:,1], 'ro-')
            mag_x,mag_y = np.sqrt(np.square(pred[i, 0]) + np.square(pred[i,1]))
            x_dir.append(np.arccos((mag_y/ mag_x)))
            y_dir.append(np.arcsin((mag_y/ mag_x)))
            # print('angles: ', (x_dir , y_dir))
            # print((gt[i,3,0], gt[i,3,1]))

            circle = pat.Arc(xy=(gt[i, 3, 0], gt[i, 3, 1]), width=x_dir[i], height=y_dir[i],
                             angle=y_dir[i], color='red')
            ax.add_patch(circle)
        # py.arrow(x= gt[0,3,0], y= gt[0,3,1], dx=-x_dir[i], dy= y_dir[i],
        #          head_width=0.5, head_length=0.3,
        #          fc='k', ec='k')
        # py.arrow(x=gt[1, 3, 0], y=gt[1, 3, 1], dx=-x_dir[i], dy=y_dir[i],
        #          head_width=0.5, head_length=0.3,
        #          fc='k', ec='k')
        # py.arrow(x=gt[2, 3, 0], y=gt[2, 3, 1], dx=x_dir[i], dy=y_dir[i],
        #          head_width=0.5, head_length=0.3,
        #          fc='k', ec='k')

        # py.axis([0, 1322 , 0 , 1079])
        py.axis([0, 20, 0, 20])

        py.show()

    return

def main():
    # checkpoint_path = '/home/siri0005/Documents/self-growing-spatial-graph/self-growing-gru-offline_avgPool/save/stanford/bookstore/srnn_model_1_0_8_batch_4_mse_hetero.tar'
    checkpoint_path = '/home/siri0005/Documents/self-growing-spatial-graph/{0}/save/crowds/srnn_model_101_3_8_batch_4_mse.tar'
    # model = int(input('select model: 1. MTV , 2. MTVP, 3. MSGTV:'))

    model_str = 'self-growing-gru-offline_avgPool'
    # ['multi_stackLSTM_chains_offline_vel' if model == 1 else
    #                    'multi_stackLSTM_chains_offline_avgPool' if model == 2 else 'self-growing-gru-offline_avgPool']
    checkpoint_path = checkpoint_path.format(model_str)

    chkpoint = torch.load(checkpoint_path)

    chk_args = chkpoint['state_dict']

    args = parser.arg_parse()
    # args.max_len = 40
    net = lstmS.lstm_network(args) #
    # net.load_state_dict(chkpoint['state_dict'])
    # loss = chkpoint['loss']
    epoch = chkpoint['epoch']
    net.eval()
    dataloader = DataLoader(args=args,sel=3, start=6, datasets=range(7))#sel=2,
    avg_time = 0

    graph = Graph.online_graph(args)

    optimizer = torch.optim.Adagrad(net.parameters(), lr=args.learning_rate,
                                    lr_decay=args.lambda_param,
                                    weight_decay=args.decay_rate)

    chk_opt = chkpoint['optimizer_state_dict']
    optimizer.state = dict(list(chk_opt['state'].items())[0:len(optimizer.state.items())])

    # plot()

    for k in range(1):
        no_targ = 0
        loss = 0
        norm_ade = []
        norm_fde = []
        targets = []
        step_count = 0
        pseudo_count = 0
        dataloader.reset_data_pointer(valid=True, dataset_pointer=6, frame_pointer=0)  # 5380
        prev_hidden = Variable(torch.zeros(args.num_layers, args.max_len, args.human_node_rnn_size))
        prev_cell = Variable(torch.zeros(args.num_layers, args.max_len, args.human_node_rnn_size))

        for i in range(20):
            # gc.collect()
            batch,targets,_ = dataloader.next_step(targets=[])
            # args.max_len = 40
            if len(batch) == 0:
                step_count += 1
                continue
            # optimizer.load_state_dict()

            plt_targets = []
            plt_pred = []
            print()
            if step_count == 0:
                graph_step = graph.ConstructGraph(current_batch=batch, framenum=step_count)
            else:
                graph_step = graph.ConstructGraph(current_batch=batch, framenum=step_count)
                pseudo_count = step_count
                if step_count >= len(graph_step.getNodes()):
                    pseudo_count = len(graph_step.getNodes()) - 1
                    graph_step.step = pseudo_count
                    graph.linkGraph(curr_graph=graph_step, frame=pseudo_count)
                else:
                    graph.linkGraph(curr_graph=graph_step, frame=step_count)

            if step_count % args.batch_size == 0 and step_count >= args.batch_size:
                targets , _ = cr.get_node_positions(graph_step=targets,
                                      curr_graph_nodes=graph_step,
                                      step=pseudo_count, istargets=True)

                start = time.time()
                # traj = graph_step.getNodes()[pseudo_count]
                print()
                velocities = torch.zeros((args.max_len, 1)).cuda()

                for k, (i, v) in zip(range(len(targets)), targets.items()):
                    begin_traj = v[0:pseudo_count+ args.batch_size]
                    graph_step.nodes[pseudo_count][i].setTargets(begin_traj)
                    if len(begin_traj):
                        vel = \
                            np.linalg.norm(np.subtract(list(begin_traj[len(begin_traj) - 1]),
                                                       list(begin_traj[0])), ord=2) / (
                                    (len(begin_traj) * 10) / 30)
                        graph_step.nodes[pseudo_count][i].vel = vel
                        velocities[k] = vel

                pred_traj, new_hidden, new_cell, new_len,\
                exc_node, disp_len, traj_hist = net(graph_step=graph_step,
                                       velocity=velocities,
                                       prev_hidden=prev_hidden,
                                       prev_cell=prev_cell, differ=False) #retain=False

                # out_targets, common_nodes = cr.get_node_positions(graph_step=targets, max_len=new_len,
                #                                                   curr_graph_nodes=graph_step,
                #                                                   step=step_count, istargets=True)
                if len(targets):
                    for i, v in targets.items():
                        if step_count >= len(v):
                            idx_start = step_count - len(v) <= len(v) if step_count - len(v) else 0
                            targets_at = v[idx_start:idx_start + args.pred_len]
                        else:
                            targets_at = v[step_count - args.batch_size + 1:int(
                                step_count + args.pred_len / 2) + 1]

                        print('pedestrian ', i)
                        print('Ground-truth: ', targets_at)
                        if not len(targets_at):
                            continue

                        targets_at = torch.Tensor(np.stack(targets_at)).cuda()
                        # out_sh = pred_traj.size()
                        # rang = len(targets_at)
                        pred_traj_at = graph_step.nodes[step_count][i].getPrediction()

                        print('predicted traj: ', pred_traj_at)

                        if len(pred_traj_at):
                            out_sh = pred_traj_at.size()
                            if targets_at.size():
                                no_targ += 1
                                err, norm_1, norm_2 = cr.online_BP(outputs=pred_traj_at,
                                                                   target=targets_at)

                                loss += (err / out_sh[0])
                                norm_ade.append(torch.sum(norm_1, 1).sum(0))
                                norm_fde.append(torch.sum(norm_2))
                                # norm_fde += (norm_2 / out_sh[0])

                                # Plot figures *****************
                                plt_targets.append(targets_at)
                                plt_pred.append(pred_traj_at)

                    gt = torch.stack(plt_targets)
                    # gt = torch.cat((traj_hist, gt), dim=0)
                    plot(gt= gt,pred= torch.stack(plt_pred) ,hold=True)

                print('forward')
                end = time.time()
                print("took {0} seconds".format((end - start)))
                avg_time += (end - start)
                # loss.backward()
                # optimizer.step()

            print('step_count = ', step_count)

            step_count += 1
        graph.onlineGraph.delGraph(step_count)
    # print('average running time in seconds :', avg_time/20.0)

    norm_ade = torch.stack(norm_ade)
    norm_fde = torch.stack(norm_fde)

    print('MSE Loss = ' + str(torch.sqrt(torch.sum(loss)) / no_targ) +
                     ' rmse ADE = ' + str(torch.sqrt(torch.sum(norm_ade)) / (no_targ * args.pred_len))
                     + 'rmse FDE = ' + str(torch.sqrt(torch.sum(norm_fde)) / no_targ))

if __name__ == '__main__':

    gt = [
        [[11.3878461516, 5.39371147352], [10.9561822118, 5.39776869012], [10.6541647795, 5.43929549527], [10.3521473472, 5.48082230042]],
        [[11.8826496243, 6.09823520228], [11.4072089417, 6.13355685264], [10.933662445, 6.12353314105], [10.4601159484, 6.11374808926]],
        [[7.29408930429,4.03335061516],[7.29408930429,4.03335061516],[8.48932066186, 4.5822681545], [9.15333808272, 4.83548620199]] #,[7.81267533441,4.34360835478]]
        ]
    # [6.26786142976, 3.24243203891], [6.79297187827, 3.71521710214],
    # , [8.48932066186, 4.5822681545], [9.15333808272, 4.83548620199]]
    # ,[8.48932066186, 4.5822681545] , [9.15333808272,	4.83548620199], [9.81230434096,	5.09490940428]]

    pred = [[[10.0091,  6.8829],[10.5786,  6.4609],[ 9.2350,  5.8140],[10.3603,  5.2944],
        [10.5896,  5.9682],[ 9.9754,  6.3041],[10.4155,  5.8527],[10.8925,  6.1944]],
            [[10.0082, 6.8844],
             [10.5784, 6.4621],
             [9.2349, 5.8151],
             [10.3602, 5.2950],
             [10.5891, 5.9691],
             [9.9749, 6.3055],
             [10.4150, 5.8536],
             [10.8913, 6.1965]],
            [[10.7536, 6.1101],
             [10.8734, 7.0276],
             [10.1709, 5.3718],
             [11.3151, 5.7906],
             [11.1353, 5.8331],
             [11.2598, 5.8725],
             [10.2524, 5.3797],
             [10.5630, 5.5015]]]

    plot(gt, pred)

    # [[10.7536, 6.1101],
    #  [10.8734, 7.0276],
    #  [10.1709, 5.3718],
    #  [11.3151, 5.7906],
    #  [11.1353, 5.8331],
    #  [11.2598, 5.8725],
    #  [10.2524, 5.3797],
    #  [10.5630, 5.5015]]
    #
    # [[10.0108, 5.4523],
    #  [11.6548, 5.3758],
    #  [10.7219, 5.4516],
    #  [10.4501, 5.9690],
    #  [11.1942, 5.1508],
    #  [10.2356, 5.5314],
    #  [11.2787, 5.2438],
    #  [10.6601, 5.5526]]
    #
    # [[12.9399, 6.2137],
    #  [13.5882, 5.8126],
    #  [13.3871, 5.7605],
    #  [12.5574, 5.7430],
    #  [13.7473, 5.9943],
    #  [13.4330, 5.6456],
    #  [13.0081, 5.5983],
    #  [13.3917, 6.1784]]

    # main()
