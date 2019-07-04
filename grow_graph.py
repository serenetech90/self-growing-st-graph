import numpy
import torch
import torch.nn as nn
# import torch.functional as F

def init_adjecancy_matrix(graph_size,max_size, mode):
    # initialize adjecancy matrix to be universal set containing all possible edges at graph-step 1
    # which means that first spatial graph starts as fully connected, then link prediction is performed each 4 steps,
    # as we are not aware of future links and we consider these as missing links that needs prediction.
    # however, we observe that starting with fully connected local graph improves the feature representation
    # at the node level, as we are combining node features with edge initial features to obtain a sort of graph structural information
    # that relieves learning in our network as it begins from a fixed point, contrary to starting with no structural information, the
    # final outcomes would heavily rely on the initial connections which may not provide sufficient feature representation for
    # robust learning at the time the graph network is still developing the link prediction heuristic.
    # specify which GNN framework our work use
    # mode: initialization mode, start with fully-connected graph or disconnected graph
    # determined empirically
    if graph_size < max_size:
        graph_size = max_size

    if mode == 1:
        adj_mat = torch.ones((graph_size, graph_size)) # square matrix
    else:
        adj_mat = torch.zeros((graph_size, graph_size))  # square matrix

    # TODO: aggregator function and updater function of node embeddings
    # TODO link prediction is a graph transduction problem ?
    #      find best neighbors based on the similarity in the node representation, use vectors similarity measures to
    #      find closest matching nodes and link those nodes as neighbors
    #      when passing messages between similar nodes
    #      p(rho) is the global aggregate function which stacks all nodes' hidden states into stacked-LSTM unit
    #      phi is the update function which is gating-based function that use LSTM (Gate updater)
    #
    # to estimate heuristic links between pedestrians we propose a metric that generalize over two metrics:
    # 1st neighborhood-based distance metric, 2nd Jaccard similarity between nodes vector embeddings, where
    # node feature representation is an embedding of pedestrian dynamics (until now it is pedestrian positional data, later
    # we will calculate the motion direction using orthogonal projection with fixed vector)
    # our link prediction algorithm begins with graph of zero cardinality, unlike existing link prediction methods that
    # are given graphs with some links (edges) provided and from there it predicts whether a link shall be added between two nodes
    # In our case, link prediction begins after observing 1.6 seconds of each pedestrian trajectory, means graph at step 4
    # such that we have enough observation to estimate relationships between pedestrians.
    # Link prediction on graph streams can be very difficult task as it may not lead to robust relational prediction results
    # it depends on the initial links that the graph starts with, in our case, we don't see necessity to make online link establishments
    # as this comes with accuracy compromise, and in our design we need to meet real-time application requirements however, we need to preserve accuracy
    # due to the risk induced by application to autonomous cars

    return adj_mat

def distance_matrix(graph_nodes, dist_mat):
    # embedded_rep = torch.Tensor(len(graph_nodes)*(len(graph_nodes)-1))

    for i in range(len(graph_nodes)):
        for j in range(len(graph_nodes)):
            diff = torch.norm((graph_nodes[i] - graph_nodes[j]), p=2)
            dist_mat[i][j] = 0 if diff == 0 else 1 / diff
            # embedded_rep[i] = F.linear(input=dist_mat[i][j].t(), weight=layer)

    # return dist_mat

def evaluate_edges (embedded_rep):

    l = len(embedded_rep) #int((len(embedded_rep)* len(embedded_rep))/2)
    diff_1 = torch.ones((l , l)) # avoid zeros as this tensor will be used forrr weighting as
    # denominator later in update function
    # diff_2 = torch.Tensor(int(l) )
    # ind_switch = False

    for i in range(len(embedded_rep)):
        v = embedded_rep[i]
        for j in range(len(embedded_rep)):
            if j != i:
                u = embedded_rep[j]
                # if ind_switch:
                #     diff_2[j*i + j] = torch.norm((u - v), p=1)
                # else:
                diff_1[i][j] = 0 if torch.norm((u - v), p=1) == 0 else 1/torch.norm((u - v), p=1)
        # ind_switch = not ind_switch

    edge_vec = diff_1 #/ len(embedded_rep)
    # diff_2 /= len(embedded_rep)
    # edge_vec = torch.stack([diff_1, diff_2], dim=1)

    return edge_vec

def update_adjecancy_matrix(adj_mat , dist_rep, edge_vec):
    # kernel_1: product series
    # kernel_2: power series
    # kernel_3: binomial series (permutations of edges set for one node/ which link(s) is improving results)
    # kernel_4: (ours) find distance between two vectors then sum the differences and weight by number of nodes in local graph
    #           then propagate the weighted vectors to the next graph step and multiply it by the next sum of difference
    #           then weight by distance (edge) between two nodes.
    #           updater of edges use log Softmax to estimate probability of having link between two pedestrians
    #           updater of nodes use Softmax followed by linear transformation layer.
    #           then update adjecancy matrix based on rounded estimated probabilities for each node pair
    # by subtracting two vectors from each other this results in direction difference between two vectors

    # memory space N^2, complexity order O(N^2)
    # if dist_spread :
    #     edge_vec = edge_vec/dist_spread #/ (dist_rep**-1) # hard attention
    # edge_vec = dist_rep * edge_vec
    # edge_vec = edge_vec / (dist_rep**-1)
    # softLog = nn.Sigmoid() # transform to probability space
    soft = nn.Softmax()
    probs = soft(dist_rep)
    # probs = torch.abs(softLog(edge_vec))
    probs = probs.round()
    for i in range(adj_mat.size()[0]):
        for j in range(adj_mat.size()[0]):
            if j != i:
                adj_mat[i][j] = probs[i][j]
    # else:
    #     adj_mat[i][i] = 0 ()
    return adj_mat
