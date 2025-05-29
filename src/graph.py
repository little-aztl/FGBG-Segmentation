import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

class GraphHelper(object):
    SIGMA = 1
    LAMBDA = 3
    def __init__(self, img_color:np.ndarray, foreground_weight, background_weight):
        self.img_color = img_color
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight

        self.source_node = self._get_index(self.img_color.shape[0] - 1, self.img_color.shape[1] - 1) + 1
        self.sink_node = self.source_node + 1

        self._get_graph()

    def _get_index(self, row, col):
        return row * self.img_color.shape[1] + col

    def _valid(self, row, col):
        if row < 0 or row >= self.img_color.shape[0] \
              or col < 0 or col >= self.img_color.shape[1]:
                return False
        return True

    def _get_neighbors(self, row, col):
        dr = [-1, 0, 1, 0]
        dc = [0, -1, 0, 1]
        for i in range(4):
            nr, nc = row + dr[i], col + dc[i]
            if self._valid(nr, nc):
                yield nr, nc

    def _get_dist(self, c1, c2):
        color_dist = np.linalg.norm(c1 - c2)
        return -np.log(color_dist) * GraphHelper.LAMBDA

    def _add_edge(self, x, y, w):
        self.graph.add_edge(x, y, capacity=w)

    def _get_graph(self):
        self.graph = nx.DiGraph()
        for r in range(self.img_color.shape[0]):
            for c in range(self.img_color.shape[1]):
                cur_node = self._get_index(r, c)
                cur_color = self.img_color[r, c]
                for nr, nc in self._get_neighbors(r, c):
                    neighbor_node = self._get_index(nr, nc)
                    neighbor_color = self.img_color[nr, nc]
                    self._add_edge(
                        cur_node, neighbor_node,
                        self._get_dist(cur_color, neighbor_color)
                    )

                self._add_edge(
                    self.source_node, cur_node,
                    self.foreground_weight[r, c]
                )

                self._add_edge(
                    cur_node, self.sink_node,
                    self.background_weight[r, c]
                )

    def get_segmentation_result(self):
        _, (fore_nodes, back_nodes) = nx.minimum_cut(self.graph, self.source_node, self.sink_node)

        fore_nodes = np.array(list(fore_nodes))
        fore_nodes = fore_nodes[fore_nodes < self.source_node]

        self.segmentation_result = np.zeros(self.img_color.shape[0] * self.img_color.shape[1])

        self.segmentation_result[fore_nodes] = 1
        return self.segmentation_result.reshape(self.img_color.shape[0], self.img_color.shape[1])





