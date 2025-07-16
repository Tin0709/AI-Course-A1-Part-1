# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

from util import Stack
from util import Queue
from util import PriorityQueue


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()
#---------------------Question 6---------------------#
    def getCostOfActions(self, actions):
        """
        Returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        # No actions means zero cost (or treat None as invalid)
        if actions is None:
            return 999999

        cost = 0
        state = self.getStartState()

        # For each action, find the matching successor and accumulate cost
        for action in actions:
            found = False
            for succ, act, stepCost in self.getSuccessors(state):
                if act == action:
                    cost += stepCost
                    state = succ
                    found = True
                    break
            if not found:
                # illegal move
                return 999999
        return cost
#---------------------Question 6---------------------#

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


#----------------Question 1 ----------------#
def depthFirstSearch(problem):
    """
    Graph‐search DFS: expand deepest nodes first, never revisiting states.
    Returns a list of actions to reach the goal.
    """
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontier = Stack()
    frontier.push((start, []))

    # Mark the start as visited now
    visited = { start }

    while not frontier.isEmpty():
        state, path = frontier.pop()

        # Goal check
        if problem.isGoalState(state):
            return path

        # Expand successors in reverse so the first one comes off the stack first
        for succ, action, cost in reversed(problem.getSuccessors(state)):
            if succ not in visited:
                visited.add(succ)  # mark as seen _when_ we push
                frontier.push((succ, path + [action]))

    # no solution
    return []
#----------------Question 1 ----------------#



#----------------Question 2 ----------------#
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first (graph-search).
    Return a list of actions that reaches the goal.
    """
    start = problem.getStartState()
    # Immediate goal check
    if problem.isGoalState(start):
        return []

    # Frontier holds (state, path)
    frontier = Queue()
    frontier.push((start, []))
    visited = { start }

    while not frontier.isEmpty():
        state, path = frontier.pop()

        # Expand
        for succ, action, cost in problem.getSuccessors(state):
            if succ not in visited:
                # Mark as visited the moment we enqueue
                visited.add(succ)
                new_path = path + [action]
                # Check for goal before pushing (optional, but saves work)
                if problem.isGoalState(succ):
                    return new_path
                frontier.push((succ, new_path))

    # No solution found
    return []

#----------------Question 2 ----------------#



#----------------Question 3 ----------------#
def uniformCostSearch(problem):
    """
    Search the node of least total cost first (graph‐search).
    Return a list of actions that reaches the goal.
    """
    start = problem.getStartState()
    # (state, path, cost_so_far)
    frontier = PriorityQueue()
    frontier.push((start, [], 0), 0)
    # best cost we’ve seen so far for each state
    best_cost = { start: 0 }

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()
        # If this popped cost is worse than a newer one, skip it
        if cost > best_cost.get(state, float('inf')):
            continue
        # Goal check
        if problem.isGoalState(state):
            return path
        # Expand
        for succ, action, stepCost in problem.getSuccessors(state):
            new_cost = cost + stepCost
            # If this path to succ is better than any before, enqueue it
            if new_cost < best_cost.get(succ, float('inf')):
                best_cost[succ] = new_cost
                frontier.push((succ, path + [action], new_cost), new_cost)

    # No solution found
    return []

#----------------Question 3 ----------------#
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

#----------------Question 4 ----------------#
def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost + heuristic first (graph‐search).
    Return a list of actions that reaches the goal.
    """
    start = problem.getStartState()
    # Frontier holds tuples: (state, path, cost_so_far)
    frontier = PriorityQueue()
    # initial priority = heuristic(start)
    frontier.push((start, [], 0), heuristic(start, problem))
    # best known cost to each state
    best_cost = { start: 0 }

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()

        # If this entry is outdated, skip it
        if cost > best_cost.get(state, float('inf')):
            continue

        # Goal check
        if problem.isGoalState(state):
            return path

        for succ, action, stepCost in problem.getSuccessors(state):
            new_cost = cost + stepCost
            # If we’ve found a cheaper path to succ, or never saw it:
            if new_cost < best_cost.get(succ, float('inf')):
                best_cost[succ] = new_cost
                priority = new_cost + heuristic(succ, problem)
                frontier.push((succ, path + [action], new_cost), priority)

    # No solution found
    return []
#----------------Question 4 ----------------#

#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
