import argparse

def arg_parse():
    train_args = argparse.ArgumentParser()

    train_args.add_argument('--batch_size' ,  type=int, default=4)
    train_args.add_argument('--seq_length', type=int, default=10) # 12 for sdd, 10 for crowds
    train_args.add_argument('--max_len', type=int, default=20)
    # train_args.add_argument('--lstm_size', type=int, default=256)
    train_args.add_argument('--pred_len', type=int, default=8)
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
    train_args.add_argument('--neighborhood_size', type=int, default=20) # 20 meter for social distance
    # if to set interpersonal distance it shall be much smaller as it is the distance used to avoid collisions
    # start with large social neighborhood

    train_args = train_args.parse_args()


    return train_args