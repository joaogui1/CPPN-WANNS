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

sys.path.append(os.getcwd() + "/../..")

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.tools.utils import count_occurrences, make_material_tree
from evosoro.tools.checkpointing import continue_from_checkpoint


VOXELYZE_VERSION = '_voxcad'
# Making sure to have the most up-to-date version of Voxelyze physics engine
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)

NUM_RANDOM_INDS = 1  # Number of random individuals to insert each generation
MAX_GENS = 100  # Number of generations
POPSIZE = 20  # Population size (number of individuals in the population)
IND_SIZE = (6, 6, 6)  # Bounding box dimensions (x,y,z). e.g. IND_SIZE = (6, 6, 6) -> workspace is a cube of 6x6x6 voxels
SIM_TIME = 10  # (seconds), including INIT_TIME!
INIT_TIME = 1
DT_FRAC = 0.9  # Fraction of the optimal integration step. The lower, the more stable (and slower) the simulation.

TIME_TO_TRY_AGAIN = 30  # (seconds) wait this long before assuming simulation crashed and resending
MAX_EVAL_TIME = 60  # (seconds) wait this long before giving up on evaluating this individual
SAVE_LINEAGES = False
MAX_TIME = 8  # (hours) how long to wait before autosuspending
EXTRA_GENS = 0  # extra gens to run when continuing from checkpoint

RUN_DIR = "basic_data"  # Subdirectory where results are going to be generated
RUN_NAME = "Basic"
CHECKPOINT_EVERY = 1  # How often to save an snapshot of the execution state to later resume the algorithm
SAVE_POPULATION_EVERY = 1  # How often (every x generations) we save a snapshot of the evolving population




class EvosoroEnv(gym.Env):
  """Classification as an unsupervised OpenAI Gym RL problem.
  Includes scikit-learn digits dataset, MNIST dataset
  """

  def __init__(self, trainSet, target):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.seed()
    self.viewer = None

    self.to_phenotype_mapping = GenotypeToPhenotypeMap()
    self.action_space = spaces.Box(np.array(0,dtype=np.float32),
                                   np.array(1,dtype=np.float32))
    self.observation_space = spaces.Box(shape=(3,))

    self.state = None
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
    if self.state[2] == 5:
      self.state[2] = 0
      if self.state[1] == 5:
        self.state[1] = 0
        self.state[0] += 1
      else:
        self.state[1] += 1
    else:
      self.state[2] += 1

    if self.state[0] == 6:
      #  evaluates phenotype, saves in rewards
      done = True
    obs = self.state

    return obs, reward, done, {}


# -- Data Sets ----------------------------------------------------------- -- #

def digit_raw():
  ''' 
  Converts 8x8 scikit digits to 
  [samples x pixels]  ([N X 64])
  '''  
  from sklearn import datasets
  digits = datasets.load_digits()
  z = (digits.images/16)
  z = z.reshape(-1, (64))
  return z, digits.target

def mnist_784():
  ''' 
  Converts 28x28 mnist digits to 
  [samples x pixels]  ([N X 784])
  '''  
  import mnist
  z = (mnist.train_images()/255)
  z = preprocess(z,(28,28))
  z = z.reshape(-1, (784))
  return z, mnist.train_labels()

def mnist_256():
  ''' 
  Converts 28x28 mnist digits to [16x16] 
  [samples x pixels]  ([N X 256])
  '''  
  import mnist
  z = (mnist.train_images()/255)
  z = preprocess(z,(16,16))

  z = z.reshape(-1, (256))
  return z, mnist.train_labels()


def preprocess(img,size, patchCorner=(0,0), patchDim=None, unskew=True):
  """
  Resizes, crops, and unskewes images

  """
  if patchDim == None: patchDim = size
  nImg = np.shape(img)[0]
  procImg  = np.empty((nImg,size[0],size[1]))

  # Unskew and Resize
  if unskew == True:    
    for i in range(nImg):
      procImg[i,:,:] = deskew(cv2.resize(img[i,:,:],size),size)

  # Crop
  cropImg  = np.empty((nImg,patchDim[0],patchDim[1]))
  for i in range(nImg):
    cropImg[i,:,:] = procImg[i,patchCorner[0]:patchCorner[0]+patchDim[0],\
                               patchCorner[1]:patchCorner[1]+patchDim[1]]
  procImg = cropImg

  return procImg

def deskew(image, image_shape, negated=True):
  """
  This method deskwes an image using moments
  :param image: a numpy nd array input image
  :param image_shape: a tuple denoting the image`s shape
  :param negated: a boolean flag telling whether the input image is negated

  :returns: a numpy nd array deskewd image

  source: https://github.com/vsvinayak/mnist-helper
  """
  
  # negate the image
  if not negated:
      image = 255-image
  # calculate the moments of the image
  m = cv2.moments(image)
  if abs(m['mu02']) < 1e-2:
      return image.copy()
  # caclulating the skew
  skew = m['mu11']/m['mu02']
  M = np.float32([[1, skew, -0.5*image_shape[0]*skew], [0,1,0]])
  img = cv2.warpAffine(image, M, image_shape, \
    flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)  
  return img



 
class GenotypeToPhenotypeMap(object):
  """A mapping of the relationship from genotype (networks) to phenotype (VoxCad simulation)."""

  def __init__(self):
    self.mapping = dict()
    self.dependencies = dict()

  def items(self):
    """to_phenotype_mapping.items() -> list of (key, value) pairs in mapping"""
    return [(key, self.mapping[key]) for key in self.mapping]

  def __contains__(self, key):
    """Return True if key is a key str in the mapping, False otherwise. Use the expression 'key in mapping'."""
    try:
      return key in self.mapping
    except TypeError:
      return False

  def __len__(self):
    """Return the number of mappings. Use the expression 'len(mapping)'."""
    return len(self.mapping)

  def __getitem__(self, key):
    """Return mapping for node with name 'key'.  Use the expression 'mapping[key]'."""
    return self.mapping[key]

  def __deepcopy__(self, memo):
    """Override deepcopy to apply to class level attributes"""
    cls = self.__class__
    new = cls.__new__(cls)
    new.__dict__.update(deepcopy(self.__dict__, memo))
    return new

  def add_map(self, name, tag, func=sigmoid, output_type=float, dependency_order=None, params=None, param_tags=None,
              env_kws=None, logging_stats=np.mean):
    """Add an association between a genotype output and a VoxCad parameter.

    Parameters
    ----------
    name : str
        A network output node name from the genotype.

    tag : str
        The tag used in parsing the resulting output from a VoxCad simulation.
        If this is None then the attribute is calculated outside of VoxCad (in Python only).

    func : func
        Specifies relationship between attributes and xml tag.

    output_type : type
        The output type

    dependency_order : list
        Order of operations

    params : list
        Constants dictating parameters of the mapping

    param_tags : list
        Tags for any constants associated with the mapping

    env_kws : dict
        Specifies which function of the output state to use (on top of func) to set an Env attribute

    logging_stats : func or list
        One or more functions (statistics) of the output to be logged as additional column(s) in logging

    """
    if (dependency_order is not None) and not isinstance(dependency_order, list):
        dependency_order = [dependency_order]

    if params is not None:
        assert (param_tags is not None)
        if not isinstance(params, list):
            params = [params]

    if param_tags is not None:
        assert (params is not None)
        if not isinstance(param_tags, list):
            param_tags = [param_tags]
        param_tags = [xml_format(t) for t in param_tags]

    if (env_kws is not None) and not isinstance(env_kws, dict):
        env_kws = {env_kws: np.mean}

    if (logging_stats is not None) and not isinstance(logging_stats, list):
        logging_stats = [logging_stats]

    if tag is not None:
        tag = xml_format(tag)

    self.mapping[name] = {"tag": tag,
                          "func": func,
                          "dependency_order": dependency_order,
                          "state": None,
                          "old_state": None,
                          "output_type": output_type,
                          "params": params,
                          "param_tags": param_tags,
                          "env_kws": env_kws,
                          "logging_stats": logging_stats}

  def add_output_dependency(self, name, dependency_name, requirement, material_if_true=None, material_if_false=None):
    """Add a dependency between two genotype outputs.

    Parameters
    ----------
    name : str
        A network output node name from the genotype.

    dependency_name : str
        Another network output node name.

    requirement : bool
        Dependency must be this

    material_if_true : int
        The material if dependency meets pre-requisite

    material_if_false : int
        The material otherwise

    """
    self.dependencies[name] = {"depends_on": dependency_name,
                                "requirement": requirement,
                                "material_if_true": material_if_true,
                                "material_if_false": material_if_false,
                                "state": None}

  def get_dependency(self, name, output_bool):
    """Checks recursively if all boolean requirements were met in dependent outputs."""
    if self.dependencies[name]["depends_on"] is not None:
        dependency = self.dependencies[name]["depends_on"]
        requirement = self.dependencies[name]["requirement"]
        return np.logical_and(self.get_dependency(dependency, True) == requirement,
                              self.dependencies[name]["state"] == output_bool)
    else:
        return self.dependencies[name]["state"] == output_bool