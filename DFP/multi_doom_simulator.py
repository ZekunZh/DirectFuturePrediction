'''
Several doom simulators running otgether
'''
from __future__ import print_function
from .doom_simulator import DoomSimulator

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import logging

class MultiDoomSimulator:
    
    def __init__(self, all_args):
        
        self.num_simulators = len(all_args)
        self.simulators = []
        for args in all_args:
            self.simulators.append(DoomSimulator(args))
            
        self.resolution = self.simulators[0].resolution
        self.num_channels = self.simulators[0].num_channels
        self.num_meas = self.simulators[0].num_meas
        self.action_len = self.simulators[0].num_buttons
        self.config = self.simulators[0].config
        self.maps = self.simulators[0].maps
        self.continuous_controls = self.simulators[0].continuous_controls
        self.discrete_controls = self.simulators[0].discrete_controls


        #####################################################
        # Initialize ConvNet for preprocessing - Zekun
        #####################################################
        # self.base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        # self.convNets = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('activation_49').output)
        self.convNets = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        logging.info("ConvNets initialized...")
            
    def step(self, actions):
        """
        Action can be either the number of action or the actual list defining the action
        
        Args:
            action - action encoded either as an int (index of the action) or as a bool vector
        Returns:
            img  - image after the step
            meas - numpy array of returned additional measurements (e.g. health, ammo) after the step
            rwrd - reward after the step
            term - if the state after the step is terminal
        """
        assert(len(actions) == len(self.simulators))
        
        imgs = []
        meass = []
        rwrds = []
        terms = []
        
        for (sim,act) in zip(self.simulators, actions):
            img, meas, rwrd, term = sim.step(act)
            #####################################################
            # preprocess images and get features: Zekun
            #####################################################
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            img_feature = self.convNets.predict(x)
            img_feature = np.squeeze(img_feature, axis=0)

            # final img shape: (channel x resolution[0] x resolution[1]) = (1 x 1 x 2048)
            img = img_feature
            # logging.info("img shape: " + str(img.shape))

            imgs.append(img)
            meass.append(meas)
            rwrds.append(rwrd)
            terms.append(term)
            
        return imgs, meass, rwrds, terms
    
    def num_actions(self, nsim):
        return self.simulators[nsim].num_actions
    
    def get_random_actions(self):
        acts = []
        for sim in self.simulators:
            acts.append(sim.get_random_action())
        return acts
