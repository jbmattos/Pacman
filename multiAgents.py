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
    dist_zero = 1e-6
    ghost_min_dist = 4
    minus_inf = float("-Inf")

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """

        pacmanPos = gameState.getPacmanPosition()
        ghostStates = gameState.getGhostStates()
        ghostPos = gameState.getGhostPositions()
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
        ghostDist = []
        for ghost in ghostPos:
            ghostDist.append(manhattanDistance(ghost, pacmanPos))
        nearGhostDist = min(ghostDist)

        # CASE: ALL GHOSTS SCARED
        if all(time != 0 for time in scaredTimes) and ghostStates[ghostDist.index(nearGhostDist)].scaredTimer >= nearGhostDist:
            return self._chase(gameState)

        # CASE: NEAR GHOST SCARED
        elif ghostStates[ghostDist.index(nearGhostDist)].scaredTimer >= nearGhostDist:
            return self._chase(gameState)

        # CASE: GHOST TOO NEAR
        elif nearGhostDist <= ReflexAgent.ghost_min_dist:
            if ghostStates[ghostDist.index(nearGhostDist)].scaredTimer >= nearGhostDist:
                return self._chase(gameState)
            else:
                return self._run(gameState)

        # CASE: FIND FOOD
        else:
            return self._eat(gameState)

    def _eat(self, currentGameState):

        # Collect legal moves and successor states
        legalMoves = currentGameState.getLegalActions()
        if Directions.STOP in legalMoves: legalMoves.remove(Directions.STOP)

        scores = []
        for action in legalMoves:
            scores.append(self._eatEvaluationFunction(currentGameState, action))

        # Choose one of the best actions
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def _eatEvaluationFunction(self, currentGameState, action):

        # Useful information you can extract from a GameState (pacman.py)
        currentPos = currentGameState.getPacmanPosition()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        # Construct food distances (current and successor game)
        allFood = newFood.asList()
        currentFoodDist = []
        successorFoodDist = []
        if not allFood:
            return 1.0
        for food in allFood:
            currentFoodDist.append(manhattanDistance(food, currentPos))
            successorFoodDist.append(manhattanDistance(food, newPos))

        #  Define min distances (current and successor game)
        if len(currentFoodDist) == 1: # Min positition
            currentMin = currentFoodDist[0]
        else:
            currentMin = min(currentFoodDist)
        if len(successorFoodDist) == 1:  # Min positition
            successorMin = successorFoodDist[0]
        else:
            successorMin = min(successorFoodDist)

        # Get directions of Pacman movement
        currentDirection = currentGameState.getPacmanState().getDirection()
        successorDirection = successorGameState.getPacmanState().getDirection()

        # ===== PRIORITY CASES =====

        # CASE: NEXT POSITION HAS FOOD
        if currentGameState.hasFood(newPos[0], newPos[1]):
            return float("Inf")

        # CASE: SAME FOOD DISTANCE AFTER ACTION
        if successorMin == currentMin:
            if self._isCorner(successorGameState, newPos):  # Successor position is a corner
                return float("-Inf")
            if successorDirection == Directions.REVERSE[currentDirection]:  # Successor position is opposite direction of movement (returning)
                return float("-Inf")

        # CASE: GREATER DISTANCE AFTER SMALLER ONE
        if currentMin > successorMin and successorDirection == Directions.REVERSE[currentDirection]:
            return float("-Inf")

        # ===== SCORE EVALUATION OF COMMON CASES =====
        minFoodDist = successorMin

        ghostPos = successorGameState.getGhostPositions()
        ghostDist = []
        for index, ghost in enumerate(newGhostStates):
            ghostDist.append(manhattanDistance(ghostPos[index], newPos))
        if len(ghostDist) == 1:
            minGhostDist = ghostDist[0]
        else:
            minGhostDist = min(ghostDist)

        score = 1.0/minFoodDist - 1.0/(100*minGhostDist)

        return score  # successorGameState.getScore()

    def _chase(self, currentGameState):
        # Collect legal moves and successor states
        legalMoves = currentGameState.getLegalActions()
        if Directions.STOP in legalMoves: legalMoves.remove(Directions.STOP)

        scores = []
        for action in legalMoves:
            scores.append(self.chaseEvaluationFunction(currentGameState, action))

        # Choose one of the best actions
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def chaseEvaluationFunction(self, currentGameState, action):

        # Pacman and ghost states
        currentPos = currentGameState.getPacmanPosition()
        floatCurrentPos = tuple(float(pos) for pos in currentPos)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        floatNewPos = tuple(float(pos) for pos in newPos)
        ghostPos = currentGameState.getGhostPositions()

        # Construct food distances (current and successor game)
        currentGhostDist = []
        successorGhostDist = []
        for ghost in ghostPos:
            currentGhostDist.append(manhattanDistance(ghost, currentPos))
            successorGhostDist.append(manhattanDistance(ghost, newPos))

        #  Define min distances (current and successor game)
        if len(ghostPos) == 1:  # Min positition
            currentMin = currentGhostDist[0]
            successorMin = successorGhostDist[0]
        else:
            currentMin = min(currentGhostDist)
            successorMin = min(successorGhostDist)
        nearGhostIdx = currentGhostDist.index(currentMin)
        nearGhostPos = ghostPos[nearGhostIdx]

        # Get directions of Pacman/nearGhost movements
        currentPacmanDir = currentGameState.getPacmanState().getDirection()
        successorPacmanDir = successorGameState.getPacmanState().getDirection()

        # ===== PRIORITY CASES =====

        # CASE: NEXT POSITION HAS A GHOST
        if nearGhostPos[0]-0.5 <= floatNewPos[0] <= nearGhostPos[0]+0.5 and nearGhostPos[1]-0.5 <= floatNewPos[1] <= nearGhostPos[1]+0.5:
            return float("Inf")

        # CASE: NEWPOS_DIST > CURRENTPOS_DIST AND PACMAN/GHOST HAVE SAME(X OR Y) AND PACMAN IS IN A CORNER
        if successorMin > currentMin and \
                (floatCurrentPos[0] == nearGhostPos[0] == floatNewPos[0] or floatCurrentPos[1] == nearGhostPos[1] == floatNewPos[1]) \
                and self._isCorner(currentGameState, currentPos):
            return float("-Inf")

        # CASE: HAS WALL BETWEEN PACMAN AND GHOST
        if currentGhostDist[nearGhostIdx] <= 5:
            hasWallBtw = self._hasWallBtw(currentGameState, newPos, nearGhostPos)
            if hasWallBtw[0]:
                return 1.0/(successorGhostDist[nearGhostIdx] + hasWallBtw[1])

        # CASE: RETURNING TO A CORNER
        if self._isCorner(successorGameState, newPos) and successorPacmanDir == Directions.REVERSE[currentPacmanDir]:
            return float("-Inf")

        # ===== SCORE EVALUATION OF COMMON CASES =====
        score = 1.0/successorGhostDist[nearGhostIdx]
        return score

    def _run(self, currentGameState):

        legalMoves = currentGameState.getLegalActions()
        if Directions.STOP in legalMoves: legalMoves.remove(Directions.STOP)

        scores = []
        for action in legalMoves:
            successorGameState = currentGameState.generatePacmanSuccessor(action)
            newPos = successorGameState.getPacmanPosition()
            ghostPos = successorGameState.getGhostPositions()
            newGhostDist = [manhattanDistance(ghost, newPos) for ghost in ghostPos]
            # scores.append(min(newGhostDist))
            scores.append(min(newGhostDist) + (sum(newGhostDist) / 100))

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def _isCorner(self, gameState, pos):

        over = False
        under = False
        right = False
        left = False

        if gameState.hasWall(pos[0], pos[1]+1):
            over = True
        if gameState.hasWall(pos[0], pos[1]-1):
            under = True
        if gameState.hasWall(pos[0]+1, pos[1]):
            right = True
        if gameState.hasWall(pos[0]-1, pos[1]):
            left = True

        if over and right:
            return True
        elif over and left:
            return True
        elif under and right:
            return True
        elif under and left:
            return True
        return False

    def _nearGap(self, gameState, pacmanPos, x, y, side):

        maxWall = max(gameState.getWalls().asList())
        maxX = maxWall[0]
        maxY = maxWall[1]
        dist = float("Inf")

        if side == "left" or side == "right":
            for y_plus in range(y,maxY):
                if not gameState.hasWall(x, y_plus):
                    d = manhattanDistance(pacmanPos, (x,y_plus))
                    if d < dist:
                        dist = d
                    break
            for y_minus in range(y,0,-1):
                if not gameState.hasWall(x, y_minus):
                    d = manhattanDistance(pacmanPos, (x,y_minus))
                    if d < dist:
                        dist = d
                    break
        elif side == "under" or side == "over":
            for x_plus in range(x,maxX):
                if not gameState.hasWall(x_plus, y):
                    d = manhattanDistance(pacmanPos, (x_plus,y))
                    if d < dist:
                        dist = d
                    break
            for x_minus in range(x,0,-1):
                if not gameState.hasWall(x_minus, y):
                    d = manhattanDistance(pacmanPos, (x_minus,y))
                    if d < dist:
                        dist = d
                    break
        return dist

    def _hasWallBtw(self, gameState, pacmanPos, ghostPos):

        if ghostPos[0] < pacmanPos[0] and gameState.hasWall(pacmanPos[0]-1, pacmanPos[1]):  # left
            nearGapDist = self._nearGap(gameState, pacmanPos, pacmanPos[0]-1, pacmanPos[1], "left")
            return (True, nearGapDist)
        elif ghostPos[0] > pacmanPos[0] and gameState.hasWall(pacmanPos[0]+1, pacmanPos[1]):  # right
            nearGapDist = self._nearGap(gameState, pacmanPos, pacmanPos[0]+1, pacmanPos[1], "right")
            return (True, nearGapDist)
        elif ghostPos[1] < pacmanPos[1] and gameState.hasWall(pacmanPos[0], pacmanPos[1]-1):  # under
            nearGapDist = self._nearGap(gameState, pacmanPos, pacmanPos[0], pacmanPos[1]-1, "under")
            return (True, nearGapDist)
        elif ghostPos[1] > pacmanPos[1] and gameState.hasWall(pacmanPos[0], pacmanPos[1]+1):  # over
            nearGapDist = self._nearGap(gameState, pacmanPos, pacmanPos[0], pacmanPos[1]+1, "over")
            return (True, nearGapDist)
        return (False, None)

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
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.minimaxSearch(gameState, deepness=0, agentIndex=self.index)

    def minimaxSearch(self, gameState, deepness, agentIndex):
   
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            deepness += 1

        if deepness >= self.depth or gameState.isWin() or gameState.isLose():    # TERMINAL NODE
            return self.evaluationFunction(gameState)
        elif agentIndex == self.index:      # MAX NODE
            return self.maxValue(gameState, deepness, agentIndex)
        else:       # MIN NODE
            return self.minValue(gameState, deepness, agentIndex)

    def minValue(self, gameState, deepness, agentIndex):

        legalActions = gameState.getLegalActions(agentIndex)
        if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)

        minValue = []
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            minValue.append(self.minimaxSearch(successor, deepness, agentIndex+1))

        return min(minValue)

    def maxValue(self, gameState, deepness, agentIndex):

        maxValue = float("-Inf")
        bestAction = None
        legalActions = gameState.getLegalActions(agentIndex)
        
        if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.minimaxSearch(successor, deepness, agentIndex+1)
            if value > maxValue:
                bestAction = action
                maxValue = value

        if deepness == 0:   # ROOT NODE
            return bestAction
        else:
            return maxValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

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
        # util.raiseNotDefined()
        return self.expectmaxSearch(gameState, deepness=0, agentIndex=self.index)

    def expectmaxSearch(self, gameState, deepness, agentIndex):

        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            deepness += 1

        if deepness >= self.depth or gameState.isWin() or gameState.isLose():    # TERMINAL NODE
            return self.evaluationFunction(gameState)
        elif agentIndex == self.index:      # MAX NODE
            return self.maxValue(gameState, deepness, agentIndex)
        else:       # EXPECT NODE
            return self.expectValue(gameState, deepness, agentIndex)

    def expectValue(self, gameState, deepness, agentIndex):

        legalActions = gameState.getLegalActions(agentIndex)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        probability = 1.0 / len(legalActions)
        expectValue = []
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.expectmaxSearch(successor, deepness, agentIndex+1)
            expectValue.append(value * probability)

        return sum(expectValue)

    def maxValue(self, gameState, deepness, agentIndex):

        maxValue = float("-Inf")
        bestAction = None
        legalActions = gameState.getLegalActions(agentIndex)

        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.expectmaxSearch(successor, deepness, agentIndex+1)
            if value > maxValue:
                bestAction = action
                maxValue = value

        if deepness == 0:   # ROOT NODE
            return bestAction
        else:
            return maxValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

