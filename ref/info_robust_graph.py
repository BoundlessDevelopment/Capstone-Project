'''given the communication and the observation graph, compute how kappa info robust the graph is'''
from typing import List

import numpy as np


def is_info_robust_graph(graph_com, graph_obs, kappa=1):
    '''returns true if the graphs are kappa robust'''
    size, _ = graph_com.shape

    for root_i in range(size):
        mask = 1 - graph_obs[root_i, :]

        for set_id in range(1, 2**size):
            set_s_str = np.binary_repr(set_id, width=size)
            set_s = np.array([i * int(j) for i, j in zip(mask, set_s_str)])
            if set_s.any():
                set_s_comp = np.array([1 - j for j in set_s])
                in_degree_from_s_comp_to_s = set_s * graph_com.dot(set_s_comp)

                if max(in_degree_from_s_comp_to_s) >= kappa:
                    continue
                print(root_i, set_s)
                return 0

    print('{0}-info robust'.format(kappa))
    return 1


def get_out_neighbor(node, graph) -> set:
    '''get the out neighbors'''
    size, _ = graph.shape

    answer = set([node])
    for j in range(size):
        if graph[j, node]:
            answer.add(j)
    return answer


def get_in_neighbors_in_set(node, graph, set_s: set) -> set:
    '''get the in neighbors that are also in the set set_s'''
    answer = set()
    for index, val in enumerate(graph[node]):
        if val and (index in set_s):
            answer.add(index)
    return answer


def get_node_i_kappa_robust(node, graph_com, graph_obs) -> int:
    '''given a reachable set determine if all other nodes are reachable'''

    size, _ = graph_com.shape
    set_s = get_out_neighbor(node, graph_obs)
    set_s.add(node)
    bar_s = set(range(size)).difference(set_s)

    minmax = size + 1
    while bar_s:
        current_node = None
        current_max = -1
        for i in bar_s:
            num_connections = len(get_in_neighbors_in_set(i, graph_com, set_s))
            if num_connections > current_max:
                current_max = num_connections
                current_node = i
        minmax = min(minmax, current_max)
        set_s.add(current_node)
        bar_s.remove(current_node)
    return minmax


def get_k_info_robust(graph_com, graph_obs):
    '''given an undirected Adj matrix graph_com and graph_obs, compute kappa'''
    size, _ = graph_com.shape
    return min(
        get_node_i_kappa_robust(i, graph_com, graph_obs) for i in range(size))


def adj_list_to_adj_matrix(adj_list):
    '''Given an adjacency list, return an adjacency matrix
        adj_list[i][j] is the jth element that connects to node i.
    '''
    size = len(adj_list)
    answer = np.zeros([size, size])
    for row_num, row in enumerate(adj_list):
        for col_num in row:
            answer[row_num, col_num] = 1

    return answer

def adj_matrix_to_adj_in_set(adj_mtx, self_loop=False):
    ''' Given an adjacency matrix return a list of set of adj nodes '''
    answer = []

    for r_num, row in enumerate(adj_mtx):
        answer.append(set())
        for c_num, val in enumerate(row):
            if val:
                answer[-1].add(c_num)
            elif self_loop and r_num == c_num:
                answer[-1].add(c_num)

    return answer

def xy_to_index(row, col, size):
    '''Given the x,y position in an array return the flatten index'''
    return (size * row) + col


def remove_nodes_from_adj_matrix(adj_mtx, nodes):
    '''Given a adj_matrix remove nodes'''
    nodes.sort(reverse=True)

    for node in nodes:
        adj_mtx = np.delete(adj_mtx, node, 0)
        adj_mtx = np.delete(adj_mtx, node, 1)

    return adj_mtx

def remove_rows_from_vector(vector, nodes):
    '''Given a adj_matrix remove nodes'''
    nodes.sort(reverse=True)

    for node in nodes:
        vector = np.delete(vector, node, 0)

    return vector

def isqrt(n):
    '''Computes the integer square root'''
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def get_corners(adj_mtx, corner_size=0):
    '''Get the corner elements from the square grid'''
    num_nodes, _ = adj_mtx.shape
    size = isqrt(num_nodes)
    end = size - 1

    corners = []
    for row in range(corner_size):
        for col in range(corner_size):
            corners.append(xy_to_index(row, col, size))
            corners.append(xy_to_index(row, end - col, size))
            corners.append(xy_to_index(end - row, col, size))
            corners.append(xy_to_index(end - row, end - col, size))

    return corners

def grid_l_inf_to_adj_matrix(size, ball):
    '''Create an adj matrix for the cross pattern and with square corner not connected'''
    answer = np.zeros([size**2, size**2], dtype=int)
    end = size - 1
    for row in range(size):
        for col in range(size):
            index = xy_to_index(row, col, size)
            for del_x in range(-ball, ball + 1):
                for del_y in range(-ball, ball + 1):
                    new_x, new_y = row + del_x, col + del_y
                    if new_x < 0 or new_x > end or new_y < 0 or new_y > end:
                        continue
                    new_index = xy_to_index(new_x, new_y, size)
                    answer[index, new_index] = 1

            answer[index, index] = 0

    return answer

def grid_l_one_to_adj_matrix(size, ball):
    '''Create an adj matrix for the cross pattern and with square corner not connected'''
    answer = np.zeros([size**2, size**2], dtype=int)
    end = size - 1

    for row in range(size):
        for col in range(size):
            index = xy_to_index(row, col, size)

            for del_x in range(-ball, ball + 1):
                for del_y in range(abs(del_x) - ball, ball - abs(del_x) + 1):
                    new_x, new_y = row + del_x, col + del_y
                    if new_x < 0 or new_x > end or new_y < 0 or new_y > end:
                        continue
                    new_index = xy_to_index(new_x, new_y, size)
                    answer[index, new_index] = 1

            answer[index, index] = 0

    return answer

def grid_node_has_left(size, corner_size):
    base = [1]*size
    base[0] = 0

    corner_base = [1]*(size - 2*corner_size)
    corner_base[0] = 0

    answer = []
    for i in range(corner_size):
        answer.extend(corner_base)
    for i in range(size - 2*corner_size):
        answer.extend(base)
    for i in range(corner_size):
        answer.extend(corner_base)

    return answer

def grid_node_has_right(size, corner_size):
    base = [1]*size
    base[-1] = 0

    corner_base = [1]*(size - 2*corner_size)
    corner_base[-1] = 0

    answer = []
    for i in range(corner_size):
        answer.extend(corner_base)
    for i in range(size - 2*corner_size):
        answer.extend(base)
    for i in range(corner_size):
        answer.extend(corner_base)
        
    return answer


def grid_node_has_up(size, corner_size):
    answer = []
    for i in range(corner_size):
        answer.extend( [1]*(size - 2*corner_size) )
    for i in range(size - 2*corner_size - 1):
        answer.extend( [1]*(size) )
    answer.extend([0]*corner_size)
    answer.extend([1]*(size - 2*corner_size))
    answer.extend([0]*corner_size)
    for i in range(corner_size - 2):
        answer.extend( [1]*(size - 2*corner_size) )
    answer.extend( [0]*(size - 2*corner_size) )
    return answer

def grid_node_has_down(size, corner_size):
    answer = [0]*(size - 2*corner_size)
    for i in range(corner_size - 2):
        answer.extend( [1]*(size - 2*corner_size) )
    answer.extend([0]*corner_size)
    answer.extend([1]*(size - 2*corner_size))
    answer.extend([0]*corner_size)
    for i in range(size - 2*corner_size - 1):
        answer.extend( [1]*(size) )
    for i in range(corner_size):
        answer.extend( [1]*(size - 2*corner_size) )
    return answer

def laplacian_from_adj_mtx(adj_mtx: np.ndarray, dim_action: int) -> np.ndarray:
    '''Construct the Laplacian given the dimensions of agents actions'''
    laplacian = np.diag(np.sum(adj_mtx, 1)) - adj_mtx
    return np.kron(laplacian, np.eye(dim_action))

def get_random_r_local_set(adj_mtx, r_local=0):
    '''Given an adj matrix find a random local set'''
    g_in = adj_matrix_to_adj_in_set(adj_mtx)
    nodes = set()
    c_nodes = set(range(len(adj_mtx)))
    all_nodes = set(range(len(adj_mtx)))

    for node in np.random.permutation(len(adj_mtx)):
        nodes.add(node)
        c_nodes.remove(node)

        for n in all_nodes:
            if len(g_in[n].intersection(nodes)) > r_local:
                c_nodes.add(node)
                nodes.remove(node)
                break

    return nodes

def main():
    '''run this function if this file is directly called'''
    graph = grid_l_inf_to_adj_matrix(10, 2)
    corners = get_corners(graph, 1)
    graph = remove_nodes_from_adj_matrix(graph, corners)
    k_info = get_k_info_robust(graph, graph)
    print("k-info robust: ", k_info)

    print('finding {0}-local set'.format((k_info - 1)//2))
    nodes = get_random_r_local_set(graph, (k_info - 1)//2)
    print("# {0}: {1}".format(len(nodes), nodes))

if __name__ == '__main__':
    main()
