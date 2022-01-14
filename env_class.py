#environment source code goes here
import sys
import numpy as np
import math

#And then for the agent
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import defects

sys.path.insert(0, "/Users/rama/Github/FerroSim")
#The most important one...
from ferrosim import Ferro2DSim

# Environment Class
class FS_Env():
    def __init__(self, N=10, k=22.5, T=300, dep_alpha=0.35,
                 time_vec=None, appliedE=None, 
                 init_pmat=None, defect_list=None, nDefects=5,
                 num_defect_moves = 3, 
                 defect_pos = None, defect_start = False,
                 mode = 'cyclic', sim_timesteps_init = 500, 
                 sim_timesteps = 50, verbose = False,
                 reward_freq = 'final', reward_type ='maxcurl',
                 POr_vec = None, TargetPattern=None):

        # Note that the defect list should simply be a list of defect objects
        # defect_start: True = randomize the positions of the defects at start of sim. False = keep them same 
        # num_defect_moves: Number of times to move each defect in a simulation

        self.N = N
        self.k = k
        self.dep_alpha = dep_alpha
        self.time_vec = time_vec
        self.nDefects = nDefects
        self.defect_pos = defect_pos
        self.defect_start = defect_start
        self.sim_timesteps = sim_timesteps
        self.sim_timesteps_init = sim_timesteps_init
        self.appliedE = appliedE
        self.verbose = verbose
        self.reward_freq = reward_freq
        self.reward_type = reward_type
        self.POr_vec = POr_vec
        self.TargetPattern = TargetPattern
        self.init_pmat = init_pmat

        if time_vec is None: 
          self.time_vec = np.linspace(0, 2, self.sim_timesteps)
          self.time_vec_init = np.linspace(0, 2, self.sim_timesteps_init)
          
        if appliedE is None:
            self.Ec_frac = 8.5
            self.appliedE = np.zeros((len(self.time_vec),2))
            #self.appliedE[:len(self.time_vec)//4,:] = -self.Ec_frac
            
            self.appliedE_init = np.zeros((self.sim_timesteps_init,2))
            #self.appliedE_init[:len(self.time_vec_init)//4,:] = -self.Ec_frac

        if defect_list is None:
            if self.defect_start:
              self.random_positions = [int(m) for m in np.random.choice(np.arange(0, self.N * self.N -1), self.nDefects)]
            else:
              self.random_positions = [int(m) for m in np.linspace(0, self.N*self.N -1, self.nDefects)]
           
            # Random fields in x and y directions
            Ef = np.array((np.random.normal(loc=15.05, scale=0.05, size=len(self.random_positions)),
                        np.random.normal(loc=-12.05, scale=0.0205, size=len(self.random_positions))))

            self.defects = [defects.LatticeDefect(Exy, self.pix_to_xy(xy), int(idx)) for Exy, xy, idx in \
                            zip(Ef.T, self.random_positions, np.arange(len(Ef.T)))]
            if self.verbose: 
              print("The defects are \n{}".format(self.defects))
            defect_list = self.defect_objs_to_list(self.defects)

        self.defect_list = defect_list

        self.defect_locations = [self.xy_to_pixind(defect.loc[0], defect.loc[1]) \
                                 for defect in self.defects]

        self.defect_chosen_idx = 0

        self.sim = None
        self.state = None
        self.done = False
        self.total_steps = self.nDefects * num_defect_moves  # steps per episode.
        self.step_number = 0  # start from 0
        self.prev_def_xy = None  # Start out with no selected defect

    def reset(self):
        
        self.state = None
        self.done = False
        self.step_number = 0
        self.time_vec = np.linspace(0, 2, self.sim_timesteps)
        if self.defect_start: 
            self.random_positions = [int(m) for m in np.random.choice(np.arange(0, self.N * self.N -1), self.nDefects)]

        # Random fields in x and y directions
        Ef = np.array((np.random.normal(loc=2.05, scale=2.05, size=len(self.random_positions)),
                        np.random.normal(loc=-1.05, scale=1.0205, size=len(self.random_positions))))

        self.defects = [defects.LatticeDefect(Exy, self.pix_to_xy(xy), int(idx)) for Exy, xy, idx in \
                        zip(Ef.T, self.random_positions, np.arange(len(Ef.T)))]

        defect_list = self.defect_objs_to_list(self.defects)

        self.defect_list = defect_list
        self.sim = Ferro2DSim(n=self.N, time_vec=self.time_vec_init, appliedE=self.appliedE_init,
                      defects=self.defect_list, k=self.k, 
                      dep_alpha=self.dep_alpha, 
                      mode='squareelectric', init = 'random') #must change to an actual pmat.
        self.step_number = 0
        self.state = self.get_reset_state()
        self.done = False
        self.reward = None
        
        return self.state, self.reward, self.done

    def get_reset_state(self):
        # the state is the polarization at the last time step and the defect list,
        # transformed to (x,y,Ex,Ey, idx) tuple
        Pmat = self.sim.getPmat(time_step=0)

        list_defects_xy = np.array([[defect.loc[0], defect.loc[1], defect.Ef[0], defect.Ef[1], defect.idx] \
                                    for defect in self.defects])
        self.defect_list_xy = list_defects_xy
        state = [Pmat, list_defects_xy, self.step_number]

        return state


    def step(self, action=None, reset = False):
       
        done = False
        #print('internal step number {} out of total {}'.format(self.step_number, self.total_steps))
        if self.step_number >= self.total_steps:
            done = True
            
        #Redo the logic here
        #-> If not done, run the sim to next step. Update the initial_p, reuse on next cycle
        #if done, then return the state, reward.
        # The only things that change on each step are the time vector, defect positions, electric fields
        
        if not done:
            
            self.action = action
            # Now we need to perform the action
            self.update_action(action)
            defect_list = self.defect_objs_to_list()  # create defect list for sending to simulator
            self.defect_list = defect_list
            initial_p = self.sim.getPmat(time_step=len(self.sim.time_vec) - 1)  # NxNx2
            initial_p = np.transpose(initial_p, (1, 2, 0))
            if self.step_number>=1:
                self.time_vec = np.linspace(0, 2, self.sim_timesteps)
                applied_field = np.zeros((len(self.time_vec), 2))
                #applied_field[:len(self.time_vec) // 4, :] = -self.Ec_frac
                self.appliedE = applied_field
            else:
                self.time_vec = np.linspace(0, 2, self.sim_timesteps_init)
                applied_field = np.zeros((len(self.time_vec), 2))
                #applied_field[:len(self.time_vec) // 4, :] = -self.Ec_frac
                self.appliedE = applied_field

            self.sim.Eloc = [(Ex * self.sim.Ec, Ey * self.sim.Ec) for (Ex, Ey) in self.defect_list]
            self.sim.appliedE = self.appliedE
            self.sim.runSimtoNextState(self.time_vec, start_p = initial_p,
                                       verbose = False)
            
        state = self.get_state()
        reward = self.get_reward()
        self.step_number += 1
        self.state = state
        if self.verbose: 
          print('reward is {} for step {}'.format(reward, self.step_number-1))
        return state, reward, done

    def update_action(self, action_tuple):
        # will return 0 if could not perform action
        # will return 1 if successful in performing action
        act_out = None
        if action_tuple is not None:

            # We get the (x,y) location of the defect
            num_moves = 4 #four ways to move, i.e. Up, down, left or right

            #Not moving is a special move handled separately.
            idx = math.floor((action_tuple-1)/(num_moves)) 
            mov_dir = (action_tuple-1)%num_moves
            mov_dir_text = ['Left', 'Up', 'Right', 'Down', 'Stay']

            if action_tuple==0: #in case we wish to 'stay'
              idx = 0
              mov_dir = 4

            if self.verbose:
              print("choosing to move defect {} {}".format(idx, mov_dir_text[int(mov_dir)]))

            defect_idx = [defect.idx for defect in self.defects]
            self.defect_chosen_idx = defect_idx.index(idx)
            row_d, col_d = self.defects[self.defect_chosen_idx].loc

            # Save internally for plotting
            self.prev_def_xy = (row_d, col_d)

            # Now we need to decipher the move
            # TODO: tunneling through to next available site when current move not available
            # Probably needs a flagged option

            if mov_dir == 0:
                new_col_d = (col_d - 1) % self.N  # Left
                new_row_d = row_d
            elif mov_dir == 1:
                new_row_d = (row_d - 1) % self.N  # Up
                new_col_d = col_d
            elif mov_dir == 2:
                new_col_d = (col_d + 1) % self.N # Right
                new_row_d = row_d
            elif mov_dir == 3:
                new_row_d = (row_d + 1) % self.N # Down
                new_col_d = col_d
            elif mov_dir == 4:  # Stay
                new_row_d = row_d
                new_col_d = col_d

            # Keep this for plotting later
            self.new_def_xy = (new_row_d, new_col_d)

            # If the chosen site is already occupied then we cannot move
            # First let's convert this new location to a pixel index
            new_pixind_loc = self.xy_to_pixind(new_row_d, new_col_d)

            # Make sure that the defect locations in pixind are updated...
            self.defect_locations = [self.xy_to_pixind(defect.loc[0], defect.loc[1]) \
                                 for defect in self.defects]

            if new_pixind_loc not in self.defect_locations:
                # Action is now valid
                # only thing to change is defect position
                self.defects[idx].update_location((new_row_d, new_col_d))
                act_out = 1
            else:
                if mov_dir!=4:
                    if self.verbose:
                        print("Tried to move defect {} from location {} to location {}, \
                        but site is occupied by another defect. Defect locations are {}".format(self.defect_chosen_idx, 
                        (row_d, col_d), (new_row_d, new_col_d), self.defects))
                        print('new_pixind_loc is {} and defect locations are {}'.format(new_pixind_loc, self.defect_locations))
                    act_out = 1
                else:
                    act_out = 0
            
            #Update defect location ix pixind form
            self.defect_locations = [self.xy_to_pixind(defect.loc[0], defect.loc[1]) \
                                 for defect in self.defects]
            
        return act_out

    def get_reward(self, mode = 'maxcurl'):
        """
        mode : (str) Default = 'maxcurl': Type of reward
        
        Several options for mode:
        ['maxcurl', 'mincurl', 'maxPmean', 'MinPmean', 'TargetPattern', 'MaxPOr']
        TargetPattern requires giving the target P distribution, reward is correlation between final state and this target. not yet implemented.
        MaxPOr is the polarization in a given orientation, which requires giving a vector m=[mx,my] and will return P_avg(Px mx, Py my). Not yet implemented.
        Curl and mean polarization are just that!

        """
        if self.reward_freq == 'final':
          if self.step_number < self.total_steps:
            return 0

        Pmat = self.get_state()[0]

        if self.reward_type == 'maxcurl' or self.reward_type == 'mincurl':
          curl_val = self.sim.calc_curl(Pmat)
          curl_mag = np.sum(np.abs(curl_val))
          pmag = np.sqrt(np.sum(np.square(np.mean(Pmat, axis =(1,2)))))
          reward_val = curl_mag  / pmag
          if self.reward_type == 'mincurl': reward_val*=-1

        if self.reward_type == 'maxPmean' or self.reward_type == 'minPmean':
          reward_val = np.sqrt(np.sum(np.square(np.mean(Pmat, axis =(1,2)))))
          if self.reward_type =='minPmean': reward_val*=-1

        if self.reward_type =='MaxPOr' or self.reward_type =='MinPOr':
          reward_val = np.dot(self.POr_vec,np.mean(Pmat, axis =(1,2))) *20 
          print('reward_val is {}'.format(reward_val))
          if self.reward_type =='MinPOr': reward_val*=-1
        
        if self.reward_type == 'TargetPattern':
          
          #Here we want to compare the correlation between the Pmat and the targetpattern
          px_corr = np.sum(correlate2d(self.TargetPattern[0,:,:], Pmat[0,:,:]), axis=(0,1))
          py_corr = np.sum(correlate2d(self.TargetPattern[1,:,:], Pmat[1,:,:]), axis=(0,1))
          corr = np.sqrt(px_corr**2 + py_corr**2)

        if self.reward_freq == 'final':
          if self.step_number >= self.total_steps:
            reward = reward_val
        else:
            reward = reward_val

        return reward

    def get_state(self):

        # the state is the polarization at the last time step and the defect list,
        # transformed to (x,y,Ex,Ey, idx) tuple
        Pmat = self.sim.getPmat(time_step=len(self.time_vec) - 1)

        list_defects_xy = np.array([[defect.loc[0], defect.loc[1], defect.Ef[0], defect.Ef[1], defect.idx] \
                                    for defect in self.defects])
        self.defect_list_xy = list_defects_xy
        state = [Pmat, list_defects_xy, self.step_number]

        return state

    def render_state(self, state=None, no_hist =False):

        # OK now the defect rendering should be changed
        # Otherwise it PROBABLY works now, needs testing...

        # plot as a quiver
        if state is None: state = self.state
        fig, axes = plt.subplots(figsize=(6, 6))
        Pvals = state[0]
        axes.quiver(Pvals[0, :, :], Pvals[1, :, :])

        # We actually want to plot the text number ID of each defect so we know which one we want to move

        axes.plot(np.array(state[1])[:, 0], np.array(state[1])[:, 1], 'ro')

        # Plot text labels
        for ind in range(len(state[1])):
            axes.text(np.array(state[1])[ind, 0] - 0.4, np.array(state[1],dtype = object)[ind, 1],
                      str(int(np.array(state[1], dtype=object)[ind, -1])), fontsize=10.0)

        # Let's get the history
        if not no_hist:
          if len(self.defects[self.defect_chosen_idx].loc_history) > 1:
              old_pos = self.defects[self.defect_chosen_idx].loc_history[-2]
              new_pos = self.defects[self.defect_chosen_idx].loc_history[-1]

              axes.plot(old_pos[0], old_pos[1], 'go', alpha=0.5)
              axes.plot(new_pos[0], new_pos[1], 'bo', alpha=1.0)

        reward = self.get_reward()
        axes.set_title('Pmap, Reward is {:.2f}'.format(reward))
        axes.set_xlim(-2, self.N + 1)
        axes.set_ylim(-2, self.N + 1)
        axes.axis('tight')

        return fig, axes

    def pix_to_xy(self, pixind):
        col = int(pixind % (self.N))
        row = int(np.floor(pixind / self.N))
        return row, col

    def xy_to_pixind(self, row, col):
        return row * self.N + col

    def defect_objs_to_list(self, defect_objs=None):
        if defect_objs is None:
            defect_objs = self.defects

        # Given a list of defect objects return the defect list that goes into the simulation
        defect_list = [(0, 0) for _ in range(self.N * self.N)]
        for defect in defect_objs:
            defect_loc_ind = self.xy_to_pixind(defect.loc[0], defect.loc[1])
            defect_list[defect_loc_ind] = defect.Ef
        return defect_list