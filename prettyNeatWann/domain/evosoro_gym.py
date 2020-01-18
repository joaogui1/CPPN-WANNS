import os
import sys
import time
import tempfile
import subprocess as sub
from functools import partial
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys

# sys.path.append(os.getcwd() + "/../..")

from domain.evosoro_base import Sim, Env, ObjectiveDict
from domain.read_write_voxelyze import *


VOXELYZE_VERSION = '_voxcad'
# Making sure to have the most up-to-date version of Voxelyze physics engine
# sub.call("cp ./" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)

# NUM_RANDOM_INDS = 1  # Number of random individuals to insert each generation
# MAX_GENS = 100  # Number of generations
# POPSIZE = 20  # Population size (number of individuals in the population)
IND_SIZE = (6, 6, 6)  # Bounding box dimensions (x,y,z). e.g. IND_SIZE = (6, 6, 6) -> workspace is a cube of 6x6x6 voxels
SIM_TIME = 10  # (seconds), including INIT_TIME!
INIT_TIME = 1
DT_FRAC = 0.9  # Fraction of the optimal integration step. The lower, the more stable (and slower) the simulation.

# TIME_TO_TRY_AGAIN = 30  # (seconds) wait this long before assuming simulation crashed and resending
# MAX_EVAL_TIME = 60  # (seconds) wait this long before giving up on evaluating this individual
# SAVE_LINEAGES = False
# MAX_TIME = 8  # (hours) how long to wait before autosuspending
# EXTRA_GENS = 0  # extra gens to run when continuing from checkpoint

RUN_DIR = "basic_data"  # Subdirectory where results are going to be generated
RUN_NAME = "Basic"
# CHECKPOINT_EVERY = 1  # How often to save an snapshot of the execution state to later resume the algorithm
# SAVE_POPULATION_EVERY = 1  # How often (every x generations) we save a snapshot of the evolving population




class EvosoroEnv(gym.Env):
  """Classification as an unsupervised OpenAI Gym RL problem.
  Includes scikit-learn digits dataset, MNIST dataset
  """

  def __init__(self, orig_size=[6, 6, 6]):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    _, self.id = tempfile.mkstemp(suffix=".vxa", prefix="Basic--id_", dir=os.path.abspath(os.getcwd()))
    self.id = self.id.partition("id_")[2]
    self.id = self.id[:self.id.index('.')]
    print("My id is ", self.id)
    self.viewer = None
    self.orig_size = orig_size
    self.phenotype = [[] for i in range(orig_size[2])]

    self.action_space = spaces.Box(low=0.0, high=1.0, shape=(5,))
    self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0]), 
                                        high=np.array([orig_size[0], orig_size[1], orig_size[2], np.sum(np.square(orig_size))]))

    self.state = [0, 0, 0]
  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self, id=1):
    ''' Initialize State'''
    # self.id = np.random.randint(low=0, high=1000000)
    self.my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, 
                      fitness_eval_init_time=INIT_TIME)
    self.my_env = Env(sticky_floor=0, time_between_traces=0)

    self.state = [0, 0, 0, 0]
    self.phenotype = [[] for i in range(self.orig_size[2])]
    return self.state

  def step(self, action):
    '''
    Generates voxel for position
    '''
    # Default values
    reward = 0
    done = False

    # ---- Writes action to voxel in position state
    if action[0] >= 0.5:
      self.phenotype[self.state[2]].append(str(np.argmax(action[1:])))
    else:
      self.phenotype[self.state[2]].append('0')

    self.state[0] += 1
    self.state[1] += self.state[0] // self.orig_size[0]
    self.state[2] += self.state[1] // self.orig_size[1]

    self.state[0] %= self.orig_size[0]
    self.state[1] %= self.orig_size[1]
    # print("action:", action, "z: ", self.state[2], "append: ", str(np.argmax(action[1:])))
    # print([len(self.phenotype[i]) for i in range(self.orig_size[2])], sep="\n\n")

    if self.state[2] == self.orig_size[2]:
      #  TODO: Test validity before evaluating
      total_voxels = np.sum([[1 if j != '0' else 0 for j in self.phenotype[i]] for i in range(self.orig_size[2])])
      if total_voxels < 1/8 * np.prod(self.orig_size):
        # print(f"Individual {self.id} has no fitness")
        return self.state, 0.0, True, {}
      # print(total_voxels)

      write_voxelyze_file(self.my_sim, self.my_env, self, RUN_DIR, RUN_NAME)
      sub.Popen(f"./voxelyze  -f " + RUN_DIR + f"/voxelyzeFiles/" + RUN_NAME + f"--id_{self.id}.vxa",
                      shell=True)
      
      evaluating = True
      init_time = time.time()
      while evaluating:
        ls_check = sub.check_output(["ls", RUN_DIR + "/fitnessFiles/"], encoding='utf-8').split()
        if f"softbotsOutput--id_{self.id}.xml" in ls_check:
          evaluating = False
        time.sleep(1)
        if time.time() - init_time > 20:
          # print(f"took too long {self.id}")
          return self.state, 0.0, True, {}

      time.sleep(2) #weird behaviors
      reward = read_voxlyze_results(RUN_DIR + f"/fitnessFiles/softbotsOutput--id_{self.id}.xml")
      # print(f"Individual {self.id} has fitness {reward}")
      done = True
      sub.Popen(f"rm  -f " + RUN_DIR + f"/voxelyzeFiles/Basic--id_{self.id}.vxa", shell=True)
      sub.Popen(f"rm  -f " + RUN_DIR + f"/fitnessFiles/softbotsOutput--id_{self.id}.xml", shell=True)

    self.state[3] = np.sum(np.square(self.state[:-1]))
    obs = self.state

    return obs, reward, done, {}
