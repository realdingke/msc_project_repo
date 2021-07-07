import numpy as np
import matplotlib.pyplot as plt

class GridWorldenv(object):
    def __init__(self, p=0.55, curr_state=20):
        #shape of the grid
        self.shape = (6,6)
        #Locations of the obstacles
        self.obstacle_locs = [(1,1),(2,5),(2,3),(3,1),(4,1),(4,2),(4,4)]
        #all possible actions
        self.action_names = ['N','E','S','W'] # Action 0 is 'N', 1 is 'E' and so on
        #Number of actions
        self.action_size = len(self.action_names)
        #action probabilty p
        self.p = p
        #all locations
        self.locs = self.get_all_locs()
        #number of states
        self.num_state = len(self.locs)
        #Locations for the terminal/absorbing states
        self.absorbing_locs = [(4,3),(1,3)]
        self.absorbing_states = []
        for loc in self.absorbing_locs:
            self.absorbing_states.append(self.loc_to_state(loc, self.locs))
        #current location/state
        self.curr_state = curr_state
        self.curr_loc = self.locs[self.curr_state]
        
        #for visualisation of the gridworld:
        # Placing the walls on a bitmap
        self.walls = np.zeros(self.shape);
        for ob in self.obstacle_locs:
            self.walls[ob] = -150
            
        # Placing the absorbers on a grid for illustration
        self.absorbers = np.zeros(self.shape)
        for ab in self.absorbing_locs:
            self.absorbers[ab] = -1
        
        # Placing the terminal states on a grid for illustration
        self.rewarders = np.zeros(self.shape)
        for i, rew in enumerate(self.absorbing_locs):
            self.rewarders[rew] = [-100,10][i]
    
    
    def get_all_locs(self):
        #get all valid locs
        locs = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.is_location((i, j)):
                    locs.append((i, j))
        return locs
    
    def loc_to_state(self,loc,locs):
        #takes list of locations and gives index corresponding to input loc
        return locs.index(tuple(loc))


    def is_location(self, loc):
        # It is a valid location if it is in grid and not obstacle
        if(loc[0]<0 or loc[1]<0 or loc[0]>self.shape[0]-1 or loc[1]>self.shape[1]-1):
            return False
        elif(loc in self.obstacle_locs):
            return False
        else:
             return True
    
    def env_start(self):
        #start the environment and randomly pick a starting loc(1/27)
        locs_available = self.locs.copy()
        locs_available.remove(self.absorbing_locs[0])
        locs_available.remove(self.absorbing_locs[1])
        self.curr_loc = locs_available[np.random.choice(range(len(locs_available)))]
        self.curr_state = self.loc_to_state(self.curr_loc, self.locs)
        
    
    def env_step(self, action):
        #env decides the action [n,e,s,w], with non-deterministic transition
        possible_next_locs = [(self.curr_loc[0]-1,self.curr_loc[1]),
                              (self.curr_loc[0],self.curr_loc[1]+1),
                              (self.curr_loc[0]+1,self.curr_loc[1]),
                              (self.curr_loc[0],self.curr_loc[1]-1)]
        if action == 0:
            if np.random.uniform(0,1)<self.p:
                next_loc = possible_next_locs[0]
            else:
                next_loc = possible_next_locs[np.random.choice([1,2,3])]
        elif action == 1:
            if np.random.uniform(0,1)<self.p:
                next_loc = possible_next_locs[1]
            else:
                next_loc = possible_next_locs[np.random.choice([0,2,3])]
        elif action == 2:
            if np.random.uniform(0,1)<self.p:
                next_loc = possible_next_locs[2]
            else:
                next_loc = possible_next_locs[np.random.choice([0,1,3])]
        elif action == 3:
            if np.random.uniform(0,1)<self.p:
                next_loc = possible_next_locs[3]
            else:
                next_loc = possible_next_locs[np.random.choice([0,1,2])]
        else:
            raise Exception(self.action_names[action]+'is not in available actions!')
        
        #check validity of the move
        if self.is_location(next_loc):
            next_position = next_loc
        else:
            next_position = self.curr_loc
        
        #reward function
        if next_position == self.absorbing_locs[0]:
            reward = -100
            done = True
        elif next_position == self.absorbing_locs[1]:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        
        next_state = self.loc_to_state(next_position, self.locs)
        
        return next_state, reward, done
    
    #drawing functions
    def draw_deterministic_policy(self, Policy):
        # Draw a deterministic policy
        # The policy needs to be a np array of 22 values between 0 and 3 with
        # 0 -> N, 1->E, 2->S, 3->W
        plt.figure()
        
        plt.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
        #plt.hold('on')
        for state, action in enumerate(Policy):
            if state in self.absorbing_states: # If it is an absorbing state, don't plot any action
                continue
            arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
            action_arrow = arrows[action] # Take the corresponding action
            location = self.locs[state] # Compute its location on graph
            plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    
        plt.show()
    
    def draw_value(self, Value):
        # Draw a policy value function
        # The value need to be a np array of 22 values 
        plt.figure()
        
        plt.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
        for state, value in enumerate(Value):
            if state in self.absorbing_states: # If it is an absorbing state, don't plot any value
                continue
            location = self.locs[state] # Compute the value location on graph
            plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    
        plt.show()
        

class MC_agent(object):
    def __init__(self, epsilon=0.1, step_size=0.2):
        self.agent_env = GridWorldenv()
        self.actions = list(range(self.agent_env.action_size)) #[0,1,2,3]
        self.step_size = step_size
        self.epsilon = epsilon
        self.discount = 0.55
        self.samples = []
        self.qvalues = np.zeros((self.agent_env.num_state,len(self.actions)))
    
    def argmax(self, q_values):
        #argmax with random tie-breaking
        top=float("-inf")
        ties=[]

        for i in self.actions:
            if q_values[i]>top:
                top=q_values[i]
                ties=[]

            if q_values[i]==top:
                ties.append(i)

        return np.random.choice(ties)
    
    def choose_action(self, state):
        #agent choose an action given the current state, according to epsilon-greedy
        if np.random.uniform(0,1)<self.epsilon:
            #take random action
            action = np.random.choice(self.actions)
        else:
            #take action according to the q function
            action = self.argmax(self.qvalues[state])
        return int(action)
    
    def update(self):
        #update the q function at the end of episode
        R = 0
        visited = []
        for tup in self.samples[::-1]:
            state, action = (tup[0],tup[1])
            if (state,action) not in visited:
                visited_state.append((state,action))
                R = self.discount*(tup[2] + R)
                self.value_table[state][action] = (self.value_table[state][action] + self.step_size*(R - self.value_table[state][action]))
    
    def play(self, num_episode=10000):
        self.total_reward = np.zeros(num_episode)
        #start to generate a trace
        for i in range(num_episode):
            self.agent_env.env_start()
            state = self.agent_env.curr_state
            action = self.choose_action(state)
            
            cumulative_reward = 0
            while True:
                next_state, reward, done = self.agent_env.env_step(action)
                self.agent_env = GridWorldenv(next_state)
                self.samples.append((next_state,action,reward))
                cumulative_reward += reward
                
                action = self.choose_action(next_state)
                
                #now update the q function at the end of each episode
                if done:
                    print(f"episode : {i}")
                    self.update()
                    self.samples = []
                    self.total_reward[i] = cumulative_reward
                    break


agent0 = MC_agent()
agent0.play()
plt.plot(list(range(len(agent0.total_reward))),agent0.total_reward)
plt.xlabel('episode number')
plt.ylabel('sum of rewards per episode')
plt.show()