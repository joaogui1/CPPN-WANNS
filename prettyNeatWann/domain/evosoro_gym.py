import os
import sys
import subprocess as sub
from functools import partial
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
import cv2

# sys.path.append(os.getcwd() + "/../..")

from domain.evosoro_base import Sim, Env, ObjectiveDict
from domain.read_write_voxelyze import *


VOXELYZE_VERSION = '_voxcad'
# Making sure to have the most up-to-date version of Voxelyze physics engine
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)

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

  def __init__(self, id, orig_size=[6, 6, 6]):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.seed()
    self.viewer = None
    self.id = id
    self.orig_size = orig_size
    self.phenotype = [[]]*orig_size[2]

    self.action_space = spaces.Box(np.array(0,dtype=np.float32),
                                   np.array(1,dtype=np.float32))
    self.observation_space = spaces.Box(shape=(3,))

    self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=make_material_tree,
                                          dependency_order=["shape", "muscleOrTissue", "muscleType", "tissueType"], output_type=int)

    self.state = [0, 0, 0]
    self.trainOrder = None
    self.currIndx = None

  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    ''' Initialize State'''
    self.my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, 
                      fitness_eval_init_time=INIT_TIME)
    self.my_env = Env(sticky_floor=0, time_between_traces=0)

    self.trainOrder = np.random.permutation(len(self.target))
    self.state = [0, 0, 0]
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
      self.phenotype[self.state[2]].append(np.argmax(action[1:]))
    else:
      self.phenotype[self.state[2]].append(0)
    self.state[0] += 1
    self.state[1] += self.state[0] // self.orig_size[0]
    self.state[2] += self.state[1] // self.orig_size[1]

    self.state[0] %= self.orig_size[0]
    self.state[1] %= self.orig_size[1]

    if self.state[2] == 6:
      #  TODO: Test validity before evaluating
      write_voxelyze_file(self.my_sim, self.my_env, self, RUN_DIR, RUN_NAME)
      sub.Popen(f"./voxelyze  -f " + RUN_DIR + "/voxelyzeFiles/" + RUN_NAME + "--id_{self.id:05}.vxa",
                      shell=True)
      reward = read_voxlyze_results()
      done = True
    obs = self.state

    return obs, reward, done, {}
