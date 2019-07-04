''' class online graph is to construct and adapt underlying graph as soon as new step is take by pedestrian '''

import numpy as np
import torch
import torch.cuda

import criterion as cr
import grow_graph as sg

batch_size = 4

class online_graph():

    def reset_graph(self,framenum):
        del self.nodes
        del self.edges
        self.onlineGraph.delGraph(framenum)

        self.nodes = [{}]
        self.edges = [[]]

    def __init__(self, args):

        self.diff = args.seq_length
        self.batch_size = args.batch_size  # 1
        # self.seq_length = args.seq_length # 1
        self.nodes = [{}]
        self.edges = [{}]
        self.onlineGraph = Graph()

    def ConstructGraph(self, current_batch, framenum, stateful=True, valid=False):
        self.onlineGraph.step = framenum
        if valid :
            for pedID, pos in current_batch.items():
                # (pedID, pos), = item.items()
                # [(pedID, pos)] = item.items()
                # if self.node
                # if pedID not in self.nodes[framenum]:
                # node_type = 'H'
                node_id = pedID
                node_pos_list = {}
                node_pos_list[framenum] = pos

                node = Node(node_id, node_pos_list)
                node.setTargets(seq=pos)
                self.onlineGraph.setNodes(framenum, node)
        else:
            for idx in range(len(current_batch)):
                try:
                    frame = current_batch[framenum]  # * self.diff
                except KeyError:
                    key = list(current_batch.keys())
                    frame = current_batch[key[0]]
                # Add nodes
                # self.nodes = self.onlineGraph.getNodes()
                # nodelist = []
                # if framenum >=1:
                #     self.nodes.append({})
                # if isinstance(frame , dict):
                #     f_inst = frame
                # else:
                #     f_inst = frame.items()

                for item in frame:
                    (pedID, pos), = item.items()
                    # [(pedID, pos)] = item.items()
                    # if self.node
                    # if pedID not in self.nodes[framenum]:
                    # node_type = 'H'
                    node_id = pedID
                    node_pos_list = {}
                    node_pos_list[framenum] = pos

                    node = Node(node_id, node_pos_list)
                    node.setTargets(seq=pos)
                    self.onlineGraph.setNodes(framenum, node)

        self.onlineGraph.dist_mat = torch.zeros(len(self.nodes), len(self.nodes))

        return self.onlineGraph

    def linkGraph(self, curr_graph, frame):

        common_nodes, n1, _ = cr.get_common_nodes(curr_graph, frame)
        # bring nodes from previous frame
        if len(common_nodes):
            for item in common_nodes:
                edge_id = (item, item)
                dist = torch.norm(
                    torch.from_numpy(np.subtract(n1[frame][item].pos[frame],
                                                 n1[frame - 1][item].pos[frame - 1])), p=2)

                edge_pos_list = dist #{frame:dist}

                curr_graph.setEdges(Edge(edge_id, edge_pos_list), framenum=frame)
        else:
            self.onlineGraph.edges.append([])  # TODO: fix memory assignement


class Graph():
    def __init__(self):
        self.adj_mat = []
        self.dist_mat = []
        self.nodes = [{}]
        self.edges = [[]]  # dict
        self.Stateful = True
        self.step = 0
        # by default the graph is stateful and each graph segment is connected to the previous temporal segment
        # unless nodes in a graph no longer exist in the scene, then we need to disconnect and destroy variables

    def getNodes(self):
        return self.nodes

    def getEdges(self):
        return self.edges

    def setNodes(self, framenum, node):
        if len(self.nodes) <= framenum:
            self.nodes.append({})
            self.nodes[framenum][node.id] = node
        # #     # self.nodes = {framenum: node}
        else:
            self.nodes[framenum][node.id] = node
        # self.nodes = {self.nodes, {framenum: node}}

    def setEdges(self, edge, framenum):
        if len(self.edges) <= framenum - 1:
            # print("appended new empty array")
            self.edges.append([])
            self.edges[framenum - 1].append(edge)
        else:
            # self.edges[framenum-1][edge.id] = edge
            #     print("append new edge")
            self.edges[framenum - 1].append(edge)

    def delGraph(self, framenum):
        del self.nodes
        del self.edges
        self.nodes = []
        self.edges = []
        for i in range(framenum):
            self.nodes.append({})
            self.edges.append([])
        # self.edges = np.zeros((framenum))


class Node():
    def __init__(self, node_id, node_pos_list):
        self.id = node_id
        self.pos = node_pos_list
        self.state = torch.zeros(batch_size, 256)  # 256 self.human_ebedding_size
        self.cell = torch.zeros(batch_size, 256)
        self.seq = []
        self.targets = []
        self.vel = 0

    def setState(self, state, cell):
        # self.state += state
        # self.cell += cell
        self.state = state
        self.cell = cell

    def getState(self):
        return self.state, self.cell

    def setPrediction(self, seq):
        self.seq = seq

    def getPrediction(self):
        return torch.Tensor(self.seq)

    def setTargets(self,seq):
        # if self.targets:
        self.targets.append(seq)
        # else:
        #     self.targets = seq

    def getTargets(self):
        return self.targets


class Edge():
    def __init__(self, edge_id, edge_pos_list):
        self.id = edge_id
        self.dist = edge_pos_list