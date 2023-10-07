#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Running the more complex resilient simulation'''
import os

from random import random
from typing import List
from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

import info_robust_graph as irg

@dataclass
class simulation_config:
    '''A container just to hold simulation parameters'''
    step_size: float = 1/40.0
    num_iter: int = 1_000
    num_rounds: int = 1
    init_state: np.ndarray = None

def action_select_matrix(dim_action_list: List[int]) -> np.ndarray:
    '''Given the dim of each agent's action return the action select matrix, i.e., the R matrix'''
    ## HP: Is this the same R matrix in Step 3 of the proposed algorithm?
    ## HP: The line below assumes that each agent can have variable number of actions, but when we look at the code, it seems
    ###### like there are only 2 actions for every agent
    dim_action = sum(dim_action_list)
    num_agents = len(dim_action_list)
    dim_state = dim_action * num_agents

    mtx = np.zeros([dim_action, dim_state])

    row, col = 0, 0
    for dim in dim_action_list:
        mtx[row:row + dim, col:col + dim] = np.eye(dim)
        row, col = row + dim, col + dim + dim_action

    return mtx

## HP: This function is never used !?
# def action_mask(dim_action_list: List[int]) -> np.ndarray:
#     '''Returns a matrix that sets the estimate components to 0'''
#     mtx = action_select_matrix(dim_action_list)

#     ## HP: I understand we need transpose as per step 3, but why a dot with itself !?
#     return mtx.transpose().dot(mtx)

def save_plot(figure, name):
    '''Saves the figure as a pdf'''
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    figure.savefig(f"{name}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)

class PlotErrorFigure(object):
    ''' Context Manager for plotting the error figures '''
    def __init__(self, name):
        self.name = name
        self.figure = None
    def __enter__(self):
        self.figure = plt.figure()
        return self.figure
    def __exit__(self, exc_type, exc_value, exc_traceback):
        plt.yscale("log")
        plt.xlabel('Iteration')
        plt.ylabel(r'$||x_{k} - x^{*}||$')
        plt.show()
        save_plot(self.figure, self.name)
        plt.close()

class Resilient:
    """Simulation for the resilient algorithm"""
    def __init__(self, sim_config, grid_width, random_agents=None, constant_agents=None, l_inf_ball = 1, D = 1, corner_size = 1):
        ## This is just the data class simulation_config, with info about step sizes, iterations etc.
        self.sim_config = sim_config   
             
        ## Assuming a square formation, N = grid_width ** 2
        self.grid_width = grid_width

        ## HP: What is corner size?
        self.corner_size = corner_size

        ## HP: What are the two actions it can perform !?
        self.dim_action_i = 2

        ## HP: What is N here, number of agents? Why do we cut corners here?
        self.N = grid_width**2 - 4*corner_size**2

        self.dim_action_list = self.dim_action_i * np.ones(self.N, dtype=int) # [2, 2, 2, .. N times]

        self.dim_action = self.dim_action_i * self.N # 2*N

        self.dim_state = self.N * self.dim_action # 2 * N^2

        self.random_agents = random_agents if random_agents else set()
        self.constant_agents = constant_agents if constant_agents else set()

        ## This is the same D in the D-local set discussed in the paper
        self.D = D

        ## HP: What is 1_inf_ball, how is Gc structured !?
        self.Gc = irg.grid_l_inf_to_adj_matrix(grid_width, l_inf_ball)
        self.corners = irg.get_corners(self.Gc, corner_size)
        self.Gc = irg.remove_nodes_from_adj_matrix(self.Gc, self.corners)
        self.Go = self.Gc + np.eye(self.N, dtype=int)
        self.adj_list_gc = irg.adj_matrix_to_adj_in_set(self.Gc, self_loop=False)

        ## Here we build the R matrix as described in Step 3 of the paper
        self.R = action_select_matrix(self.dim_action_list)
        self.A, self.b = self.get_gradient()

        ## HP: For god's sake I need more documentation/comments!! What is F!?
        self.F = block_diag(*[self.A[2*i:2*(i+1),:] for i in range(self.N)])
        self.RF = self.R.transpose().dot(self.F)
        self.RB = self.R.transpose().dot(self.b)

        ## HP: What is NE here !?
        self.NE = -np.linalg.inv(self.A).dot(self.b)

        self.example = 'position_plot'

    def block_diag(self):
        ''' Block diag the matrix A to create F '''
        answer = np.zeros([self.dim_action, self.dim_state])

        ## HP: What is this !?
        for i in range(self.N):
            answer[self.dim_action_i*i:self.dim_action_i*(i+1),self.dim_action*i:self.dim_action*(i+1)] = self.A[self.dim_action_i*i:self.dim_action_i*(i+1),:]

        return answer

    def get_gradient(self):
        ''' Returns the gradient for the cost function '''
        local_cost = irg.grid_l_one_to_adj_matrix(self.grid_width, 1)
        local_cost = irg.remove_nodes_from_adj_matrix(local_cost, self.corners)
       
        grad_rel_position = irg.laplacian_from_adj_mtx(local_cost, self.dim_action_i)
        grad_average = ((1/self.N)**2)*np.kron(np.ones([self.N,self.N]),np.eye(self.dim_action_i))
        A = grad_average + grad_rel_position

        # L: Left, R: Right, U: Up, D: Down
        L = np.array([irg.grid_node_has_left(self.grid_width, self.corner_size)]).transpose()
        R = np.array([irg.grid_node_has_right(self.grid_width, self.corner_size)]).transpose()
        U = np.array([irg.grid_node_has_up(self.grid_width, self.corner_size)]).transpose()
        D = np.array([irg.grid_node_has_down(self.grid_width, self.corner_size)]).transpose()

        # Change in X is Right - Left
        cost_x_rel = R - L

        # Change in Y is Up - Down
        cost_y_rel = U - D

        b = np.kron(cost_x_rel, np.array([[1],[0]])) + np.kron(cost_y_rel, np.array([[0],[1]]))

        return A, b

    def remove_extreme_D_average(self, received_values, agent_value):
        '''Filter that removes the extreme messages received'''

        # TODO: Re write this function using multiprocessing library.
        # This sidesteps the issues with the GIL to allow proper 
        # parallelism that can't be done with threads in python.
        # This method is slow because this is converting everything into 
        # matrix operations but the matrix is massive. Probably faster to
        # just have each agent have there own process and compute the update
        # instead of trying to use numpy/matrix computation for speed up.
        # Especially since this part of the code can't be written as a 
        # matrix operation.

        received_values.sort()
        max_index = len(received_values)-1
        low_index = 0
        high_index = max_index

        for j_highest in range(self.D):
            if received_values[max_index - j_highest] >= agent_value:
                high_index = high_index-1
            else:
                break
        for j_lowest in range(self.D):
            if received_values[j_lowest] <= agent_value:
                low_index = low_index+1
            else:
                break

        # Note: Using numpy sum, and mean is acutally slower than doing this?!
        average = agent_value
        for index in range(low_index, high_index + 1):
            average += received_values[index]
        average = average/(high_index - low_index + 2)
        return average

    def filter_communicated_message(self, state_y):
        state_v = np.zeros([self.dim_state,1]) # [[0], [0], [0], ... 2 N^2 times], dim_state is 2N^2
        # The state is organized as follows:
        # [[a1], [a2], [b1], [b2], ... N pairs of [x1], [x2], where x are all agents in N, - This is agent 1's estimate
        #  [a1], [a2], [b1], [b2], ... N pairs of [x1], [x2], where x are all agents in N, - This is agent 2's estimate
        #  . 
        #  . 
        #  . 
        #  N times                                                                         - This is agent n's estimate]

        for agent_i in range(self.N):
            for state_j in range(self.N):
                for component_k in range(self.dim_action_i):
                    offset = self.dim_action_i*state_j+component_k
                    state_index_i = self.dim_action*agent_i + offset
                    ## If neighbour is in the observation graph, get the true value
                    if self.Go[agent_i, state_j] == 1:
                        state_index_j = self.dim_action*state_j + offset
                        state_v[state_index_i] = state_y[state_index_j,state_j]
                    ## If neighbour is not in the observation graph, prune extreme D values and estimate
                    else:
                        agent_i_in_messages = [ state_y[self.dim_action*X + offset, agent_i] for X in self.adj_list_gc[agent_i]]
                        state_v[state_index_i] = self.remove_extreme_D_average(agent_i_in_messages, state_y[state_index_i,agent_i])

        return state_v

    def adversarial_communication(self, state_x):
        dim = self.dim_action
        state_y = np.kron(state_x, np.ones([1,self.N]))

        for agent in self.random_agents:
            ## Type 1: Adversarial agents who add a bit of normal noise
            state_y[agent*dim:(agent + 1)*dim,:] = state_y[agent*dim:(agent + 1)*dim,:] + np.random.normal(0,1,[dim,self.N])
            state_y[agent*dim:(agent + 1)*dim,agent] = state_x[agent*dim:(agent+1)*dim].transpose()[0]
        for agent in self.constant_agents:
            ## Type 2: Adversarial agents that always give constant signals
            state_y[agent*dim:(agent + 1)*dim,:] = np.ones([dim,self.N])
            state_y[agent*dim:(agent + 1)*dim,agent] = state_x[agent*dim:(agent+1)*dim].transpose()[0]

        return state_y

    def iterate_algo(self, init_state):
        ''' This function is the one that runs the proposed algorithm '''
        state_x = init_state # [1.26, -3.45, some random shit, ... 2N^2 times]
        print('Compare')
        print(state_x)
        records = [np.linalg.norm(self.NE-self.R.dot(state_x),2)]
        pos_records = [self.R.dot(state_x)]
        
        for i in range(self.sim_config.num_iter):
            if (i%100) == 0:
                print(f"Iteration {i} of {self.sim_config.num_iter}")
            
            state_y = self.adversarial_communication(state_x)
            state_v = self.filter_communicated_message(state_y)
            state_x = state_v - self.sim_config.step_size*(self.RF.dot(state_v)+self.RB)

            records.append(np.linalg.norm(self.NE-self.R.dot(state_x),2))
            pos_records.append(self.R.dot(state_x))

        return records, pos_records, state_x

    def position_plot(self, pos_record, save=False, index_set = None, adversarial=None):
        figure = plt.figure()
        pos_record = np.reshape(pos_record,[-1, self.dim_action])

        indexs = range(self.N) if index_set is None else index_set
        for i in indexs:
            if i in adversarial:
                plt.plot(pos_record[:,2*i],pos_record[:,2*i+1], '--')
            else:
                plt.plot(pos_record[:,2*i],pos_record[:,2*i+1])
        
        for i in range(self.N):
            plt.plot(self.NE[2*i], self.NE[2*i+1], marker='.', markersize=3, color="red")
        
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')

        # plt.rc('axes', labelsize = 12)    # fontsize of the x and y labels
        # plt.rc('xtick', labelsize = 8)    # fontsize of the tick labels
        # plt.rc('ytick', labelsize = 8)    # fontsize of the tick labels
      
        plt.show()
        if save:
            save_plot(figure, "{0}".format(self.example))
        plt.close()

def plot_save_file_data(selected, adversarial):
    pos_records = []
    with open('position_data.txt') as f:
        for line in f:
            data = [ float(num) for num in line.split(',')]
            pos_records.append(np.array([data]).T)

    game.position_plot(pos_records, save = True, index_set=selected, adversarial=adversarial)

    err_record = []
    with open('error_data.txt') as f:
        for line in f:
            err_record.append(float(line))

    with PlotErrorFigure('error_plot') as error_fig:
        plt.plot(err_record)

def main(game, sim_config):
    '''main function to run the examples'''
    init_state = -7 + 14*np.random.rand(game.dim_state,1) # [1.26, -3.45, some random shit, ... 2N^2 times]
    for i in range(sim_config.num_rounds):
        print(f'Executing round: {i} / {sim_config.num_rounds}')
        err_record, pos_record, last_iter = game.iterate_algo(init_state)

        # Writing the results to a file because the run time can be long.
        # If the program crashes or needs to stop then the simulation can
        # be continued by using these files.
        with open('position_data.txt', 'a') as f:
            for data in pos_record:
                record = data.T
                record = list(record[0])
                f.write("%s\n" % ",".join([str(num) for num in record]))

        with open('error_data.txt', 'a') as f:
            for data in err_record:
                f.write("%s\n" % data)

        with open('last_state.txt', 'a') as f:
            data = last_iter.T
            f.write("%s\n" % ",".join([str(num) for num in data[0]]))

        print(last_iter)
        init_state = last_iter

    return err_record, pos_record, last_iter

if __name__ == "__main__":
    continue_run = True
    
    if not continue_run:
        os.remove("error_data.txt")
        os.remove("position_data.txt")
        os.remove("last_state.txt")
    else:
        # TODO: Add code here if you want to continue from where we left off.
        pass

    sim_config = simulation_config()
    
    selected = [0,1,2,3,4,5,6,7,8,9,10,11]

    ## Why are there no adversial agents LOL !?
    random_agents = []
    constant_agents = []
    adversarial = random_agents + constant_agents

    # These are the old examples I used.
    # game = Resilient(sim_config, grid_width = 4, random_agents=None, constant_agents=None, l_inf_ball=1)
    # game = Resilient(sim_config, grid_width = 10, random_agents=set([4, 6, 11, 19, 26, 32, 38, 41]), constant_agents=None, l_inf_ball=2) # Fails
    # game = Resilient(sim_config, grid_width = 10, random_agents=set([5, 71, 8, 74, 10, 78, 17, 87, 28, 95, 46, 61]), constant_agents=None, l_inf_ball=2, D=3, corner_size = 1)
    game = Resilient(sim_config, grid_width = 4, random_agents=random_agents, constant_agents=constant_agents, l_inf_ball=1)

    # These are other examples where I wanted to make Gc != Go. 
    # grid_width = 10; game = Resilient(sim_config, grid_width = grid_width, random_agents=set([0,3,6,28,31,34,37,58,61,64,67,89,92,95]), constant_agents=None, l_inf_ball=1, D=1, corner_size = 1)
    # grid_width = 4; game = Resilient(sim_config, grid_width = grid_width, random_agents=set([0,11]), constant_agents=None, l_inf_ball=1, D=1, corner_size = 1)
    # grid_width = 5; game = Resilient(sim_config, grid_width = grid_width, random_agents=set([0,7,14]), constant_agents=None, l_inf_ball=1, D=1, corner_size = 1)
    # game.Go = irg.grid_l_one_to_adj_matrix(grid_width,1)
    # game.Go[1,grid_width], game.Go[grid_width,1] = 1, 1
    # game.Go[grid_width-2,2*grid_width - 1], game.Go[2*grid_width-1, grid_width-2] = 1, 1
    # game.Go[(grid_width-2)*grid_width, (grid_width-1)*grid_width + 1], game.Go[(grid_width-1)*grid_width + 1, (grid_width-2)*grid_width] = 1,1
    # game.Go[grid_width**2 - 2, grid_width**2 - grid_width -1], game.Go[grid_width**2 - grid_width -1, grid_width**2 - 2] = 1,1
    # game.Go = irg.remove_nodes_from_adj_matrix(game.Go, irg.get_corners(game.Go, 1))
    
    main(game, sim_config)

    plot_save_file_data(selected, adversarial)
