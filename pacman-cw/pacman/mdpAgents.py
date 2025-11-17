# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print("Starting up MDPAgent!")
        name = "Pacman"

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print("Running registerInitialState for MDPAgent!")
        print("I'm at:")
        print(api.whereAmI(state))
        
    # This is what gets run in between multiple games
    def final(self, state):
        print("Looks like the game just ended!")

    # For now I just move randomly
    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Random choice between the legal options.
        return api.makeMove(random.choice(legal), legal)
        """
        current_state = self.getStateKey(state)
        legal = api.legalActions(state)
        
        # Want to keep moving
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
        # Find best action using computed value
        best_action = None
        best_value = float('-inf')
        
        # Evaluate each legal action
        for action in legal:
            # Compute expected value for action 
            q_value = 0
            transitions = self.getTransitionStates(current_state, action)
            
            # Sum all outcomes
            for next_s, prob in transitions:
                reward = self.getReward(current_state, action, next_s)
                q_value += prob * (reward + self.gamma * self.values[next_s])
                
            # Track best action 
            if q_value > best_value:
                best_value = q_value 
                best_action = action
                
        # returns the best legal move
        return api.makeMove(best_action, legal)
    
    # Getting current state 
    def getStateKey(self, state):
        current_pos = api.whereAmI(state)
        food_pos = api.food
        # Convert food list to tuple 
        food_tuple = tuple(sorted(food_pos))
        
        return (current_pos, food_tuple)
    
    # Transition function
    def getTransitionStates(self, state, action):
        """
        Returns list of (next_state, probability) tuples
        taking 'action' from 'state' 
        """
        
        transitions = []
        
        # 80% intended action 
        intended_next = self.simulateAction(state, action)
        transitions.append((intended_next, 0.8))
        
        # 10% each perpendicular direction 
        perpendicular = self.getPerpendicularActions(action)
        for perp_action in perpendicular:
            perp_next = self.simulateAction(state, perp_action)
            # 10 for each directions
            transitions.append((perp_next, 0.1))
        return transitions
    
    def getPerpendicularActions(self, action):
        # Perpendicular to vertical
        if action == Directions.NORTH or action == Directions.SOUTH:
            return[Directions.EAST, Directions.WEST]
        # when its EAST or WEST
        else: 
            return[Directions.NORTH, Directions.SOUTH]
        
    # Simulate taking action, returns state 
    def simulateAction(self, state, action):
        pacman_pos = api.whereAmI(state)
        food_list = list(api.food(state))
        walls = api.walls(state)
        
        # Get next position based on action 
        x, y = self.direction_vectors[action]
        next_pos = (pacman_pos[0] + x, pacman_pos[1] + y)
        
        # Check if wall 
        if next_pos in walls:
            next_pos = pacman_pos
            
        # Update food if eaten 
        if next_pos in food_list:
            food_list.remove(next_pos)
            
        # Return next position and update food positions
        return (next_pos, tuple(sorted(food_list)))
    
    # Calculate rward for transition 
    def getReward(self, state, action, next_state):
        current_food = set(state[1])
        next_food = set(next_state[1])
        
        # Check if food was eaten 
        if len(next_food) < len(current_food):
            #eaten all the food
            if len(next_food) == 0:
                return 510 # + 10 for food, +500 for winning
            else: 
                return 10 
        else: 
            # Time penalty for each move
            return -1
