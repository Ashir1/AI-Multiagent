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

        "Add more of your code here if you want to"

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

        "*** YOUR CODE HERE ***"
        pos = currentGameState.getPacmanPosition()
        curScore = currentGameState.getScore()
        newScore = successorGameState.getScore()

        minCurFoodDist = 1000
        for food in currentGameState.getFood().asList():
            foodDist = util.manhattanDistance(pos, food)
            if foodDist < minCurFoodDist:
                minCurFoodDist = foodDist

        foodDist = 1000
        minFoodDist = 1000
        for food in newFood.asList():
            foodDist = util.manhattanDistance(newPos, food)
            if foodDist < minFoodDist:
                minFoodDist = foodDist

        minGhostDist = 1000
        for ghost in newGhostStates:
            ghostDist = util.manhattanDistance(newPos, ghost.getPosition())
            if ghostDist < minGhostDist:
                minGhostDist = ghostDist

        netFoodDist = 0
        if foodDist != 0:
            netFoodDist = minCurFoodDist - minFoodDist

        direction = currentGameState.getPacmanState().getDirection()

        if minGhostDist <= 1 or action == Directions.STOP:
            return 0

        if newScore > curScore:
            return 1000

        if netFoodDist > 0:
            return 100

        if action == direction:
            return 10

        return 1




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
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            if depth + 1 == self.depth:  # Terminal Test
                return self.evaluationFunction(gameState)
            if not gameState.getLegalActions(0):  # NO legalActions means game finished(win or lose)
                return self.evaluationFunction(gameState)
            maxV= -10000
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                new_val = min_value(successor, depth + 1, 1)
                maxV = max(maxV, new_val)
            return maxV

        # For all ghosts.
        def min_value(gameState, depth, player):

            if not gameState.getLegalActions(player):  # NO legalActions means game finished(win or lose)
                return self.evaluationFunction(gameState)
            minV = 10000
            for action in gameState.getLegalActions(player):
                successor = gameState.generateSuccessor(player, action)
                if player == (gameState.getNumAgents() - 1):
                    new_val = max_value(successor, depth)
                else:
                    new_val = min_value(successor, depth, player + 1)
                minV = min(minV, new_val)
            return minV

        score = - 10000
        firstAction = ''
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            new_val = min_value(successor, 0, 1)
            prev_score = score
            score = max(score, new_val)
            if prev_score != score:
                firstAction = action
        return firstAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):

        def max_value(gameState, alpha, beta, depth):
            if depth + 1 == self.depth:  # Terminal Test
                return self.evaluationFunction(gameState)
            if not gameState.getLegalActions(0):  # NO legalActions means game finished(win or lose)
                return self.evaluationFunction(gameState)
            maxV = -10000
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                new_val = min_value(successor, alpha, beta, depth + 1, 1)
                maxV = max(maxV, new_val)
                if maxV > beta:
                    return maxV
                alpha = max(alpha, maxV)
            return maxV

        # For all ghosts.
        def min_value(gameState, alpha, beta, depth, player):

            if not gameState.getLegalActions(player):  # NO legalActions means game finished(win or lose)
                return self.evaluationFunction(gameState)
            minV = 10000
            for action in gameState.getLegalActions(player):
                successor = gameState.generateSuccessor(player, action)
                if player == (gameState.getNumAgents() - 1):
                    new_val = max_value(successor, alpha, beta, depth)
                else:
                    new_val = min_value(successor, alpha, beta, depth, player + 1)
                minV = min(minV, new_val)
                if minV < alpha:
                    return minV
                beta = min(beta, minV)
            return minV

        score = - 10000
        alpha = -10000
        beta = 10000
        firstAction = ''
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            new_val = min_value(successor, alpha, beta, 0, 1)
            prev_score = score
            score = max(score, new_val)
            if prev_score != score:
                firstAction = action
            if new_val > beta:
                return firstAction
            alpha = max(alpha, new_val)
        return firstAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            if depth + 1 == self.depth:  # Terminal Test
                return self.evaluationFunction(gameState)
            if not gameState.getLegalActions(0):  # NO legalActions means game finished(win or lose)
                return self.evaluationFunction(gameState)
            maxV = -10000
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                new_val = expecti_value(successor, depth + 1, 1)
                maxV = max(maxV, new_val)
            return maxV

        # For all ghosts.
        def expecti_value(gameState, depth, player):

            if not gameState.getLegalActions(player):  # NO legalActions means game finished(win or lose)
                return self.evaluationFunction(gameState)
            expectiV = 0
            for action in gameState.getLegalActions(player):
                successor = gameState.generateSuccessor(player, action)
                if player == (gameState.getNumAgents() - 1):
                    new_val = max_value(successor, depth)
                else:
                    new_val = expecti_value(successor, depth, player + 1)
                actionLength = len(gameState.getLegalActions(player))
                if actionLength == 0:
                    return 0
                expectiV += new_val / actionLength
            return expectiV


        score = - 10000
        firstAction = ''
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            new_val = expecti_value(successor, 0, 1)
            prev_score = score
            score = max(score, new_val)
            if prev_score != score:
                firstAction = action
        return firstAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I separated the function into a piecewise function of sorts with two cases:
    whether or not the ghosts were scared due to PacMan having eaten a power pellet.

    Then, it both cases I added to the existing score the reciprocal of the sum
    of all the Manhattan distances to the various food pellets, rewarding the
    state for having fewer, closer food pellets.

    In the case where the ghosts are scared, I added the total time left for the
    ghost to be scared so PacMan could act accordingly.  However, I did scale
    the weight of that down by a factor of 2 as that is not as impactful as the
    amount and distance to the food remaining for Pacman to consume.



    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    totalRecipFoodDist = 0
    for food in newFood.asList():
        foodDist = util.manhattanDistance(newPos, food)
        if foodDist != 0:
            totalRecipFoodDist += 1/foodDist

    totScared = sum(newScaredTimes)

    score = currentGameState.getScore()
    if totScared > 0:
        score += totalRecipFoodDist + 0.5*totScared
    else:
        score += totalRecipFoodDist
    return score







    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
