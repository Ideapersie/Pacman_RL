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
        
        # MDP Parameters 
        # Discount factor 
        self.gamma = 0.95 
        # Theta for convergence 
        self.theta = 0.01
        
        # Max iterations 
        self.max_iterations = 100

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
        
        # Retrieve environment information 
        pacman_pos = api.whereAmI(state)
        food_pos = api.food(state)
        # Making it iterable
        food_set = set(food_pos)
        # Capsules 
        capsules = set(api.capsules(state))
        walls = set(api.walls(state))
        # Ghosts 
        ghosts = api.ghostStatesWithTimes(state)
        
        # Map information 
        corners = api.corners(state)
        width = 0 
        height = 0 
        
        for (x,y) in corners: 
            if x >= width: width = x + 1
            if y >= height: height = y + 1
            
        # Reward function 
        rewards = {}
        
        # Liviing reward should be low, allows for finding long maze rewards
        for x in range(width):
            for y in range(height):
                if (x,y) not in walls:
                    rewards[(x,y)] = -0.05
                    
                    # Trap Check 
                    wall_count = 0 
                    
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nx, ny = x + dx, y + dy 
                        if (nx, ny) in walls: 
                            wall_count += 1
                            
                    # If 3 sides are wall = dead end trap, penalty equals to death
                    if wall_count >= 3 and len(food_set) > 1:
                        rewards[(x, y)] -= 750.0 
        
        # Food rewards in float, since living death is decimal
        for f in food_set:
            # Prevents trap reward overriding
            rewards[f] = 10.0
            
        # Capsule reward, slightly higher
        for c in capsules:
            rewards[c] = 15.0
            
        # Ghost penalty
        for ghost_pos, timer in ghosts:
            gx, gy = int(ghost_pos[0]), int(ghost_pos[1])
            
            if timer > 0: # Scared ghosts
                # Doesnt chase since timer could run it, risky 
                pass
                
            else: 
                # Instant death, heavy penalty
                rewards[(gx,gy)] = -1000.0
                
                # Area close to ghosts 
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    # Adds buffer distance, incase pacman slips
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                        rewards[(nx, ny)] -= 500.0
            
        # Value Iteration
        values = {}
        for x in range(width):
            for y in range(height):
                if (x,y) in walls:
                    values[(x, y)] = 0.0
                else:
                    values[(x,y)] = rewards.get((x,y), 0.0)
                    
        # Run iterations
        for i in range(self.max_iterations):
            new_values = values.copy()
            delta = 0 

            for x in range(width):
                for y in range(height):
                    if (x,y) in walls:
                        continue
                    
                    # Iterate over all directions 
                    possible_actions = [Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]
                    best_action_value = -99999.0
                    
                    for action in possible_actions:
                        # Calculate expected value 
                        expected_val = self.get_expected_value(x, y, action, values, walls, width, height)
                        if expected_val > best_action_value:
                            best_action_value = expected_val
                            
                    # Bellman equation update 
                    new_value = rewards[(x,y)] + self.gamma * best_action_value
                    new_values[(x,y)] = new_value
                    
                    delta = max(delta, abs(new_value - values[(x, y)]))
                    
            values = new_values
            if delta < self.theta:
                break
            
        # Select best action, after full value iteration 
        best_action = Directions.STOP
        max_utility = -99999.0
        
        px, py = pacman_pos
        
        for action in legal:
            utility = self.get_expected_value(px, py, action, values, walls, width, height)
            
            # Check best utility 
            if utility > max_utility:
                max_utility = utility
                best_action = action 
                
        return api.makeMove(best_action, legal)
    
    # Models non-determinism in api.py
    def get_expected_value(self, x, y, action, values, walls, width, height):
        # Map directions 
        vectors = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }
        
        # Define transition probability to account for slipup
        if action == Directions.NORTH:
            moves = [(0.8, vectors[Directions.NORTH]), (0.1, vectors[Directions.EAST]), (0.1, vectors[Directions.WEST])]
        
        elif action == Directions.SOUTH:
            moves = [(0.8, vectors[Directions.SOUTH]), (0.1, vectors[Directions.EAST]), (0.1, vectors[Directions.WEST])]
            
        elif action == Directions.EAST:
            moves = [(0.8, vectors[Directions.EAST]), (0.1, vectors[Directions.NORTH]), (0.1, vectors[Directions.SOUTH])]
            
        elif action == Directions.WEST:
            moves = [(0.8, vectors[Directions.WEST]), (0.1, vectors[Directions.NORTH]), (0.1, vectors[Directions.SOUTH])]
            
        else: 
            return values.get((x, y), 0.0)
        
        expected_value = 0.0
        
        # Simulation loop, checking 
        for prob, (dx, dy) in moves:
            nx, ny = int( x+dx), int (y+dy)
            
            # Wall check 
            if (nx, ny) in walls or nx < 0 or nx >= width or ny < 0 or ny >= height:
                nx, ny = x, y
                
            expected_value += prob * values.get((nx,ny), 0.0)
        return expected_value