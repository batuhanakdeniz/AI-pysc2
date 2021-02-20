# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))


class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def step(self, obs):
    super(MoveToBeacon, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not beacon:
        return FUNCTIONS.no_op()
      beacon_center = numpy.mean(beacon, axis=0).round()
      return FUNCTIONS.Move_screen("now", beacon_center)
    else:
      return FUNCTIONS.select_army("select")


class CollectCustomized(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectCustomized, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()  # Average location.
      distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      closest_mineral_xy = minerals[numpy.argmin(distances)]
      return FUNCTIONS.Move_screen("now", closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")


class FindMean(base_agent.BaseAgent):

  def __init__(self):
    super().__init__()
    self.lastmove = (0,0)
    self.printed = False
  def step(self, obs):
    super(FindMean, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      np_minerals = numpy.array(minerals).astype(float)
      mean = numpy.mean(np_minerals, axis=0)
      print(mean)
      #input()
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()
      if not self.printed:
        plt.axis('equal')
        plt.scatter(np_minerals[:, 0], np_minerals[:, 1], c='b')
        plt.scatter(mean[0], mean[1], c='r', label = 'mean')
        plt.scatter(marine_xy[0], marine_xy[1], marker = "1", c = "black", label = 'marine')
        plt.legend()
        plt.show()
        self.printed = True
      return FUNCTIONS.Move_screen("now", (0,0))#closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")


class GoClosestMean(base_agent.BaseAgent):

  def __init__(self):
    super().__init__()
    self.lastmove = (0,0)
  def step(self, obs):
    super(GoClosestMean, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()  # Average location.
      means = numpy.mean(minerals, axis = 0)
      mineral_distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      mean_distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      print(marine_xy)
      closest_mineral_xy = means[numpy.argmin(mean_distances)]
      return FUNCTIONS.Move_screen("now", closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")

class CircleAround(base_agent.BaseAgent):

  def __init__(self):
    super().__init__()
    self.lastmove = (0,0)
    self.remaining = 25
  def step(self, obs):
    super(CircleAround, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()  # Average location.
      means = numpy.mean(minerals, axis = 0)
      mineral_distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      mean_distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      if self.remaining == 0:
        if self.lastmove[0]==0 and self.lastmove[1]==0:
          self.lastmove = (0,70)
        elif self.lastmove[0]==0 and self.lastmove[1]==70:
          self.lastmove = (70,70)
        elif self.lastmove[0]==70 and self.lastmove[1]==70:
          self.lastmove = (70,0)
        elif self.lastmove[0]==70 and self.lastmove[1]==0:
          self.lastmove = (0,0)
        self.remaining = 25
      self.remaining = self.remaining - 1
      return FUNCTIONS.Move_screen("now", self.lastmove)
    else:
      return FUNCTIONS.select_army("select")

class Find3Clusters(base_agent.BaseAgent):
  """An agent specifically for find 3 clusters."""

  def __init__(self):
    super().__init__()
    self.lastmove = (0,0)
    self.printed = False
  def step(self, obs):
    super(Find3Clusters, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()
      np_minerals = numpy.array(minerals).astype(float)
      if not self.printed:
        print(type(np_minerals))
        print(np_minerals)
        centroid, label = kmeans2(np_minerals, 3)
        print(centroid)
        print(label)
        c1 = np_minerals[label==0]
        c2 = np_minerals[label==1]
        c3 = np_minerals[label==2]
        plt.axis('equal')
        plt.scatter(c1[:, 1], c1[:, 0], c='r')
        plt.scatter(c2[:, 1], c2[:, 0], c='g')
        plt.scatter(c3[:, 1], c3[:, 0], c='b')
        plt.scatter(marine_xy[1], marine_xy[0], marker = "1")
        plt.scatter(centroid[:, 1], centroid[:, 0], marker = "x")
        plt.show()
        self.printed = True
      return FUNCTIONS.Move_screen("now", (0,0))#closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")


class Find4Clusters(base_agent.BaseAgent):
  """An agent specifically for find 3 clusters."""

  def __init__(self):
    super().__init__()
    self.lastmove = (0,0)
    self.printed = False
  def step(self, obs):
    super(Find4Clusters, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()
      np_minerals = numpy.array(minerals).astype(float)
      if not self.printed:
        print(type(np_minerals))
        print(np_minerals)
        centroid, label = kmeans2(np_minerals, 4)
        print(centroid)
        print(label)
        c1 = np_minerals[label==0]
        c2 = np_minerals[label==1]
        c3 = np_minerals[label==2]
        c4 = np_minerals[label==3]
        plt.axis('equal')
        plt.scatter(c1[:, 0], c1[:, 1], c='r')
        plt.scatter(c2[:, 0], c2[:, 1], c='g')
        plt.scatter(c3[:, 0], c3[:, 1], c='b')
        plt.scatter(c3[:, 0], c3[:, 1], c='black')
        plt.scatter(marine_xy[0], marine_xy[1], marker = "1", label = 'marine')
        plt.scatter(centroid[:, 0], centroid[:, 1], marker = "x", label = 'centroids')
        plt.title('Clusters and current position')
        plt.legend()
        plt.show()
        self.printed = True
      return FUNCTIONS.Move_screen("now", (0,0))#closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")

class Go3Clusters(base_agent.BaseAgent):
  """An agent specifically for find 3 clusters."""

  def __init__(self):
    super().__init__()
    self.lastmove = (0,0)
    self.printed = False
  def step(self, obs):
    super(Go3Clusters, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()
      np_minerals = numpy.array(minerals).astype(float)
      centroids, label = kmeans2(np_minerals, 3)
      w1 = np_minerals[label==0].shape[0]
      w2 = np_minerals[label==1].shape[0]
      w3 = np_minerals[label==2].shape[0]
      cent_dis = numpy.linalg.norm(centroids - marine_xy, axis=1)
      closest_cluster_minerals = np_minerals[label==numpy.argmin(cent_dis)]
      mineraldistances = numpy.linalg.norm(closest_cluster_minerals - marine_xy, axis=1)
      closest_mineral_xy = minerals[numpy.argmin(mineraldistances)]

      if not self.printed:
        print(type(np_minerals))
        print(np_minerals)
        
        print("Distance of centroids:", cent_dis)
        print("Closest cluster:", closest_cluster_minerals)
        print("Size:", np_minerals.shape)
        #mineral_distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
        #weight1 = centroid
        print(centroids)
        print(label)
        c1 = np_minerals[label==0]
        c2 = np_minerals[label==1]
        c3 = np_minerals[label==2]
        plt.axis('equal')
        plt.scatter(c1[:, 0], c1[:, 1], c='r')
        plt.scatter(c2[:, 0], c2[:, 1], c='g')
        plt.scatter(c3[:, 0], c3[:, 1], c='b')
        plt.scatter(marine_xy[0], marine_xy[1], marker = "1", label = 'marine')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", label = 'centroids')
        plt.title('Clusters and current position')
        plt.legend()
        plt.show()
        self.printed = True
      return FUNCTIONS.Move_screen("now", closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")


class Go3Weighted(base_agent.BaseAgent):
  
  def __init__(self):
    super().__init__()
    self.lastmove = (0,0)
    self.printed = False
  def step(self, obs):
    super(Go3Weighted, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()
      np_minerals = numpy.array(minerals).astype(float)
      centroids, label = kmeans2(np_minerals, 3)
      w1 = np_minerals[label==0].shape[0]
      w2 = np_minerals[label==1].shape[0]
      w3 = np_minerals[label==2].shape[0]
      s1 = w1/numpy.linalg.norm(centroids[0] - marine_xy)
      s2 = w2/numpy.linalg.norm(centroids[1] - marine_xy)
      s3 = w3/numpy.linalg.norm(centroids[2] - marine_xy)
      scores = [s1, s2, s3]
      np_sc = numpy.array(scores)
      print("Scores:", np_sc)
      cent_dis = numpy.linalg.norm(centroids - marine_xy, axis=1)
      closest_cluster_minerals = np_minerals[label==numpy.argmin(np_sc)]
      mineraldistances = numpy.linalg.norm(closest_cluster_minerals - marine_xy, axis=1)
      closest_mineral_xy = minerals[numpy.argmin(mineraldistances)]

      if not self.printed:
        print(type(np_minerals))
        print(np_minerals)
        
        print("Distance of centroids:", cent_dis)
        print("Closest cluster:", closest_cluster_minerals)
        print("Size:", np_minerals.shape)
        #mineral_distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
        #weight1 = centroid
        print(centroids)
        print(label)
        c1 = np_minerals[label==0]
        c2 = np_minerals[label==1]
        c3 = np_minerals[label==2]
        plt.axis('equal')
        plt.scatter(c1[:, 0], c1[:, 1], c='r')
        plt.scatter(c2[:, 0], c2[:, 1], c='g')
        plt.scatter(c3[:, 0], c3[:, 1], c='b')
        plt.scatter(marine_xy[0], marine_xy[1], marker = "1", label = 'marine')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", label = 'centroids')
        plt.title('Clusters and current position')
        plt.legend()
        plt.show()
        self.printed = True
      return FUNCTIONS.Move_screen("now", closest_mineral_xy)
    else:
      return FUNCTIONS.select_army("select")
