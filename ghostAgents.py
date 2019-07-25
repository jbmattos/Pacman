# ghostAgents.py
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util

class GhostAgent( Agent ):

    TRAIL = {}
    @staticmethod
    def UpdateTRAIL(pacPos, chaseDistance):
        if GhostAgent.TRAIL:
            for pos in GhostAgent.TRAIL.keys():
                dist = manhattanDistance(pos, pacPos)
                if dist > chaseDistance:
                    del GhostAgent.TRAIL[pos]
        return

    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist

class MinimaxGhost( GhostAgent ):

    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, depth='2'):
        self.index = index
        self.initalState = False
        self.depth = int(depth)
        self.pacIndex = 0
        self.chaseDistance = 10

    def getAction( self, state ):

        # Update chase trail
        GhostAgent.UpdateTRAIL(state.getPacmanPosition(), self.chaseDistance)
        # Check if is initial state
        if state.getGhostPosition(self.index) == state.data.agentStates[self.index].start.pos:
            self.initalState = True

        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 1:
            return legalActions[0]
        elif len(legalActions) == 0:
            return Directions.STOP

        # INITIAL STATE: inside square
        if self.initalState:
            if Directions.NORTH in legalActions:
                successor = state.generateSuccessor(self.index, Directions.NORTH)
                successorActions = successor.getLegalActions(self.index)
                if Directions.NORTH not in successorActions:
                    self.initalState = False
                return Directions.NORTH


        isScared = state.getGhostState(self.index).scaredTimer > 0
        selfPos = state.getGhostPosition(self.index)
        pacPos = state.getPacmanPosition()
        distance = manhattanDistance(selfPos, pacPos)

        # CASE: CHASE PACMAN
        if distance <= self.chaseDistance and not isScared:

            if selfPos in GhostAgent.TRAIL.keys():  # another ghost passed through this position
                actionsNotChosen = [a for a in legalActions if a not in GhostAgent.TRAIL[selfPos]]  # actions not yet chosen by previous ghosts

                if len(actionsNotChosen) == 1:  # just 1 action not previously chosen
                    if actionsNotChosen[0] != Directions.REVERSE[state.getPacmanState().getDirection()]:
                        action = actionsNotChosen[0]
                        GhostAgent.TRAIL[selfPos] += [action]
                        return action
                    else:  # only action is reverse to pacman's direction
                        action = self.search(state)
                        GhostAgent.TRAIL[selfPos] = [action]
                        return action
                elif len(actionsNotChosen) > 1:  # more than 1 action not previously chosen
                    action = self.search(state, possibleActions=actionsNotChosen)
                    GhostAgent.TRAIL[selfPos] += [action]
                    return action
                elif len(actionsNotChosen) == 0:  # all legalActions already chosen
                    action = self.search(state)
                    GhostAgent.TRAIL[selfPos] = [action]
                    return action

            else:  # no previous ghost passed through this position
                action = self.search(state)
                GhostAgent.TRAIL[selfPos] = [action]
                return action

        # NORMAL BEHAVIOR: MINIMAX SEARCH
        else:
            return self.search(state)

    def search(self, state, deepness=0, possibleActions=None):

        if deepness == self.depth:
            return self.evaluationFunction(state)
        else:
            return self.moveGhost(state, deepness, possibleActions)

    def moveGhost(self, gameState, deepness, possibleActions):

        if possibleActions is not None:
            nextMoves = possibleActions
        else:
            nextMoves = gameState.getLegalActions(self.index)
        if len(nextMoves) == 0:
            return self.evaluationFunction(gameState)

        nextStates = [gameState.generateSuccessor(self.index, action) for action in nextMoves]
        scores = [self.movePacman(nextState, deepness=deepness) for nextState in nextStates]

        if deepness == 0:   # ROOT NODE
            return nextMoves[scores.index(max(scores))]
        else:
            return max(scores)

    def movePacman(self, gameState, deepness):

        nextMoves = gameState.getLegalPacmanActions()
        if len(nextMoves) == 0:
            return self.evaluationFunction(gameState)
        if Directions.STOP in nextMoves:
            nextMoves.remove(Directions.STOP)

        nextStates = [gameState.generateSuccessor(0, action) for action in nextMoves]
        scores = [self.search(nextState, deepness=(deepness+1)) for nextState in nextStates]
        return min(scores)

    def evaluationFunction(self, state):
        isScared = state.getGhostState(self.index).scaredTimer > 0
        pacPos = state.getPacmanPosition()
        selfPos = state.getGhostPosition(self.index)
        distance = manhattanDistance(selfPos, pacPos)
        if distance == 0:
            distance = 1e-6

        if isScared:
            return distance
        else:
            return 1.0/distance

class ExpectimaxGhost( GhostAgent ):

    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, depth='2'):
        self.index = index
        self.initalState = False
        self.depth = int(depth)
        self.pacIndex = 0
        self.chaseDistance = 10

    def getAction( self, state ):

        # Update chase trail
        GhostAgent.UpdateTRAIL(state.getPacmanPosition(), self.chaseDistance)
        # Check if is initial state
        if state.getGhostPosition(self.index) == state.data.agentStates[self.index].start.pos:
            self.initalState = True

        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 1:
            return legalActions[0]
        elif len(legalActions) == 0:
            return Directions.STOP

        # INITIAL STATE: inside square
        if self.initalState:
            if Directions.NORTH in legalActions:
                successor = state.generateSuccessor(self.index, Directions.NORTH)
                successorActions = successor.getLegalActions(self.index)
                if Directions.NORTH not in successorActions:
                    self.initalState = False
                return Directions.NORTH

        isScared = state.getGhostState(self.index).scaredTimer > 0
        selfPos = state.getGhostPosition(self.index)
        pacPos = state.getPacmanPosition()
        distance = manhattanDistance(selfPos, pacPos)

        # CASE: CHASE PACMAN
        if distance <= self.chaseDistance and not isScared:

            if selfPos in GhostAgent.TRAIL.keys():  # another ghost passed through this position
                actionsNotChosen = [a for a in legalActions if a not in GhostAgent.TRAIL[selfPos]]  # actions not yet chosen by previous ghosts

                if len(actionsNotChosen) == 1:  # just 1 action not previously chosen
                    if actionsNotChosen[0] != Directions.REVERSE[state.getPacmanState().getDirection()]:
                        action = actionsNotChosen[0]
                        GhostAgent.TRAIL[selfPos] += [action]
                        return action
                    else:  # only action is reverse to pacman's direction
                        action = self.search(state)
                        GhostAgent.TRAIL[selfPos] = [action]
                        return action
                elif len(actionsNotChosen) > 1:  # more than 1 action not previously chosen
                    action = self.search(state, possibleActions=actionsNotChosen)
                    GhostAgent.TRAIL[selfPos] += [action]
                    return action
                elif len(actionsNotChosen) == 0:  # all legalActions already chosen
                    action = self.search(state)
                    GhostAgent.TRAIL[selfPos] = [action]
                    return action

            else:  # no previous ghost passed through this position
                action = self.search(state)
                GhostAgent.TRAIL[selfPos] = [action]
                return action

        # NORMAL BEHAVIOR: EXPECTIMAX SEARCH
        else:
            return self.search(state)

    def search(self, state, deepness=0, possibleActions=None):

        if deepness == self.depth:
            return self.evaluationFunction(state)
        else:
            return self.moveGhost(state, deepness, possibleActions)

    def moveGhost(self, gameState, deepness, possibleActions):

        if possibleActions is not None:
            nextMoves = possibleActions
        else:
            nextMoves = gameState.getLegalActions(self.index)
        if len(nextMoves) == 0:
            return self.evaluationFunction(gameState)

        nextStates = [gameState.generateSuccessor(self.index, action) for action in nextMoves]
        scores = [self.movePacman(nextState, deepness=deepness) for nextState in nextStates]

        if deepness == 0:   # ROOT NODE
            return nextMoves[scores.index(max(scores))]
        else:
            return max(scores)

    def movePacman(self, gameState, deepness):

        nextMoves = gameState.getLegalPacmanActions()
        if len(nextMoves) == 0:
            return self.evaluationFunction(gameState)
        if Directions.STOP in nextMoves:
            nextMoves.remove(Directions.STOP)

        nextStates = [gameState.generateSuccessor(0, action) for action in nextMoves]
        scores = [self.search(nextState, deepness=(deepness+1)) for nextState in nextStates]
        return sum(scores) / len(nextMoves)

    def evaluationFunction(self, state):
        isScared = state.getGhostState(self.index).scaredTimer > 0
        pacPos = state.getPacmanPosition()
        selfPos = state.getGhostPosition(self.index)
        distance = manhattanDistance(selfPos, pacPos)
        if distance == 0:
            distance = 1e-6

        if isScared:
            return distance
        else:
            return 1.0/distance