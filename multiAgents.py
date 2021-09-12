# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if(action == 'Stop'):
            return 0
            
        foodDistances = []
        for i in range(newFood.width):
            for j in range(newFood.height):
                if(newFood[i][j] == True):
                    foodDistances.append(manhattanDistance(newPos, (i, j)))
        
        ghostDistances = []
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDistances.append(manhattanDistance(newPos, ghostPos))
        
        minGhostDistance = min(ghostDistances)
        minFoodDistance = min(foodDistances) if len(foodDistances) > 0 else 1
    
        if(newFood != currentGameState.getFood() and minGhostDistance > 1):
            return float('inf')

        return minGhostDistance/minFoodDistance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def maxValue(self, gameState, currentDepth):
        if(currentDepth == self.depth):
            return self.evaluationFunction(gameState)
        v = float('-inf')
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            v = max(v, self.minValue(gameState.generateSuccessor(0, action), 1, currentDepth))
        if(v == float('-inf')):
            return self.evaluationFunction(gameState)
        return v

    def minValue(self, gameState, currentGhostIndex, currentDepth):
        if(currentDepth == self.depth):
            return self.evaluationFunction(gameState)
        v = float('inf')
        legalActions = gameState.getLegalActions(currentGhostIndex)
        for action in legalActions:
            if(currentGhostIndex == gameState.getNumAgents() - 1):
                v = min(v, self.maxValue(gameState.generateSuccessor(currentGhostIndex, action), currentDepth + 1))
            else:
                v = min(v, self.minValue(gameState.generateSuccessor(currentGhostIndex, action), currentGhostIndex + 1, currentDepth))
        if(v == float('inf')):
            return self.evaluationFunction(gameState)
        return v

    def minimaxDecision(self, gameState):
        bestAction, bestScore = None, float('-inf')
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            score = self.minValue(gameState.generateSuccessor(0, action), 1, 0)
            if(score > bestScore):
                bestScore = score
                bestAction = action
        return bestAction

    # def minimax(self, gameState, currentDepth, currentAgentIndex):
    #     if(currentAgentIndex == gameState.getNumAgents()):
    #         currentAgentIndex = 0
    #         currentDepth += 1
        
    #     if(currentDepth == self.depth):
    #         return None, self.evaluationFunction(gameState)

    #     bestAction, bestScore = None, None
    #     if(currentDepth == 0):
    #         legalActions = gameState.getLegalActions(currentAgentIndex)
    #         for action in legalActions:
    #             _, score = self.minimax(gameState.generateSuccessor(currentAgentIndex, action), currentDepth, currentAgentIndex + 1)
    #             if(bestScore is None or score > bestScore):
    #                 bestAction = action
    #                 bestScore = score
    #     else:
    #         legalActions = gameState.getLegalActions(currentAgentIndex)
    #         for action in legalActions:
    #             _, score = self.minimax(gameState.generateSuccessor(currentAgentIndex, action), currentDepth, currentAgentIndex + 1)
    #             if(bestScore is None or score < bestScore):
    #                 bestAction = action
    #                 bestScore = score
        
    #     if(bestScore is None):
    #         return None, self.evaluationFunction(gameState)
    #     return bestAction, bestScore

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # action, score = self.minimax(gameState, 0, 0)
        action = self.minimaxDecision(gameState)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, currentDepth, alpha, beta):
        if(currentDepth == self.depth):
            return self.evaluationFunction(gameState)
        v = float('-inf')
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            v = max(v, self.minValue(gameState.generateSuccessor(0, action), 1, currentDepth, alpha, beta))
            if(v > beta):
                return v
            alpha = max(alpha, v)
        if(v == float('-inf')):
            return self.evaluationFunction(gameState)
        return v

    def minValue(self, gameState, currentGhostIndex, currentDepth, alpha, beta):
        if(currentDepth == self.depth):
            return self.evaluationFunction(gameState)
        v = float('inf')
        legalActions = gameState.getLegalActions(currentGhostIndex)
        for action in legalActions:
            if(currentGhostIndex == gameState.getNumAgents() - 1):
                v = min(v, self.maxValue(gameState.generateSuccessor(currentGhostIndex, action), currentDepth + 1, alpha, beta))
            else:
                v = min(v, self.minValue(gameState.generateSuccessor(currentGhostIndex, action), currentGhostIndex + 1, currentDepth, alpha, beta))
            if(v < alpha):
                return v
            beta = min(beta, v)
        if(v == float('inf')):
            return self.evaluationFunction(gameState)
        return v

    def alphaBetaSearch(self, gameState):
        bestAction, bestScore = None, float('-inf')
        alpha, beta = float('-inf'), float('inf')
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            score = self.minValue(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if(score > bestScore):
                bestScore = score
                bestAction = action
            if(score > beta):
                return action
            alpha = max(alpha, score)
        return bestAction

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        action = self.alphaBetaSearch(gameState)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValue(self, gameState, currentDepth):
        if(currentDepth == self.depth):
            return self.evaluationFunction(gameState)
        v = float('-inf')
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            v = max(v, self.minValue(gameState.generateSuccessor(0, action), 1, currentDepth))
        if(v == float('-inf')):
            return self.evaluationFunction(gameState)
        return v

    def minValue(self, gameState, currentGhostIndex, currentDepth):
        if(currentDepth == self.depth):
            return self.evaluationFunction(gameState)
        v = float('inf')
        legalActions = gameState.getLegalActions(currentGhostIndex)
        prob = None
        if(len(legalActions) > 0):
            prob = 1.0/len(legalActions)
        for action in legalActions:
            if(v == float('inf')):
                v = 0.0
            if(currentGhostIndex == gameState.getNumAgents() - 1):
                v += self.maxValue(gameState.generateSuccessor(currentGhostIndex, action), currentDepth + 1)*prob
            else:
                v += self.minValue(gameState.generateSuccessor(currentGhostIndex, action), currentGhostIndex + 1, currentDepth)*prob
        if(v == float('inf')):
            return self.evaluationFunction(gameState)
        return v

    def expectimaxDecision(self, gameState):
        bestAction, bestScore = None, float('-inf')
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            score = self.minValue(gameState.generateSuccessor(0, action), 1, 0)
            if(score > bestScore):
                bestScore = score
                bestAction = action
        return bestAction

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action = self.expectimaxDecision(gameState)
        return action

# def betterEvaluationFunction(currentGameState):
#     """
#     Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#     evaluation function (question 5).

#     DESCRIPTION: <write something here so we know what you did>
#     """
#     ghostStates = currentGameState.getGhostStates()
#     ghostDistances = []
#     for ghostState in ghostStates:
#         ghostPos = ghostState.getPosition()
#         ghostDistances.append(manhattanDistance(currentGameState.getPacmanPosition(), ghostPos))
#     if(currentGameState.getNumFood() == 0):
#         return min(ghostDistances)
#     return min(ghostDistances)/currentGameState.getNumFood()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    ghostStates = currentGameState.getGhostStates()
    food = currentGameState.getFood()
    ghostDistances = []
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDistances.append(manhattanDistance(currentGameState.getPacmanPosition(), ghostPos))
    foodDistances = []
    for i in range(food.width):
        for j in range(food.height):
            if(food[i][j] == True):
                foodDistances.append(manhattanDistance(currentGameState.getPacmanPosition(), (i, j)))
    minGhostDistance = min(ghostDistances)
    if(currentGameState.getNumFood() == 0):
        return (currentGameState.getScore() - minGhostDistance)
    return (currentGameState.getScore() - minGhostDistance)/currentGameState.getNumFood()

# def betterEvaluationFunction(currentGameState):
#     """
#     Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#     evaluation function (question 5).

#     DESCRIPTION: <write something here so we know what you did>
#     """
#     import numpy
#     ghostStates = currentGameState.getGhostStates()
#     food = currentGameState.getFood()
#     ghostDistances = []
#     for ghostState in ghostStates:
#         ghostPos = ghostState.getPosition()
#         ghostDistances.append(manhattanDistance(currentGameState.getPacmanPosition(), ghostPos))
#     foodDistances = []
#     for i in range(food.width):
#         for j in range(food.height):
#             if(food[i][j] == True):
#                 foodDistances.append(manhattanDistance(currentGameState.getPacmanPosition(), (i, j)))
#     minGhostDistance = numpy.mean(ghostDistances)
#     if(currentGameState.getNumFood() == 0):
#         return (currentGameState.getScore() - minGhostDistance)
#     return (currentGameState.getScore() - currentGameState.getNumFood())/(currentGameState.getNumFood()) - minGhostDistance

# Abbreviation
better = betterEvaluationFunction
