from random import randint
from BaseAI import BaseAI
import time
import math


class Timeout(Exception):
  """Subclass base exception for code clarity."""
  pass


class PlayerAI(BaseAI):

  def __init__(self):
    self.TIMELIMIT = 0.1 + 0.05 - 0.025 # the margin to use for completing the call
    self.method = 'alphabeta' # ['minimax', 'alphabeta'] this will control what strategy is used to search
    self.start_time = None

  def score(self, grid):
    smooth_wt = 0.1*0.0
    empty_wt = 2.7*0.0
    max_wt = 1.0*0.0
    monaticity_wt = 1.0
    smooth_val = self.smoothness(grid)
    max_val = grid.getMaxTile()
    empty_val = len(grid.getAvailableCells())
    monaticity_val = self.monotacity2(grid)
    heuristic_val = smooth_wt*smooth_val + max_wt*max_val + empty_wt*empty_val + monaticity_wt*monaticity_val
    #print heuristic_val
    return heuristic_val


  def smoothness(self, grid):
    directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
    #directionVectors = (DOWN_VEC, RIGHT_VEC) = ((1, 0), (0, 1))
    smooth_score = 0.0
    for x in xrange(grid.size):
      for y in xrange(grid.size):
        # if cell is occupied then find the neighbours
        if grid.map[x][y] != 0:
          for vec in directionVectors:
            for d in xrange(1, grid.size):
              x_n = x+vec[0]*d
              y_n = y+vec[1]*d
              if x_n < 0 or x_n >= grid.size or y_n < 0 or y_n >= grid.size:
                break
                # i.e. end of the direction reached without finding any neighbour
              if grid.map[x_n][y_n] != 0:
                smooth_score -= abs(math.log(grid.map[x][y], 2) - math.log(grid.map[x_n][y_n], 2))
                break
    return smooth_score

  def monotacity(self, grid):
    totals = [0.0 for i in xrange(grid.size)]

    # first find monaticity in UP/DOWN
    for x in xrange(grid.size):
      first, second = None, None
      for y in xrange(grid.size):
        if grid.map[x][y] != 0:
          first = second
          second = grid.map[x][y]
          if first and second:
            value = abs(math.log(first, 2) - math.log(second, 2))
            if first > second:
              totals[0] += value
            else:
              totals[1] += value

    # now find monaticity in LEFT/RIGHT
    for y in xrange(grid.size):
      first, second = None, None
      for x in xrange(grid.size):
        if grid.map[x][y] != 0:
          first = second
          second = grid.map[x][y]
          if first and second:
            value = abs(math.log(first, 2) - math.log(second, 2))
            if first > second:
              totals[2] += value
            else:
              totals[3] += value

    return max(totals[0], totals[1]) + max(totals[2], totals[3])

  def monotacity2(self, grid):
    totals = [0.0,0.0]

    # first find monaticity in UP/DOWN
    for x in xrange(grid.size):
      first, second = None, None
      for y in xrange(grid.size):
        if grid.map[x][y] != 0:
          first = second
          second = grid.map[x][y]
          if first and second:
            if first > second:
              totals[0] += 1
            else:
              totals[0] -= 1

    # now find monaticity in LEFT/RIGHT
    for y in xrange(grid.size):
      first, second = None, None
      for x in xrange(grid.size):
        if grid.map[x][y] != 0:
          first = second
          second = grid.map[x][y]
          if first and second:
            if first > second:
              totals[1] += 1
            else:
              totals[1] -= 1

    avg_monotacity = abs(totals[0])+abs(totals[1]) / len(grid.getAvailableCells())

    return avg_monotacity

  def forecast_move(self, grid, n_move):
    gridCopy = grid.clone()
    gridCopy.move(n_move)
    return gridCopy

  def isWin(self, grid):
    return grid.getMaxTile() >= 2048

  def getMove(self, grid):

    # store the time this call was started
    self.start_time = time.clock()

    moves = grid.getAvailableMoves()

    # no move left for Player
    if not moves:
      return None

    #pick one random move
    best_move = moves[randint(0, len(moves) - 1)]
    best_score = self.score(self.forecast_move(grid, best_move))

    try:
      depth = 1
      while True:
        #print depth
        if self.method == 'minimax':
            score, move = self.minimax(grid, depth)
        elif self.method == 'alphabeta':
            score, move = self.alphabeta(grid, depth)
        else:
            raise NotImplementedError

        if best_score < score:
            best_move = move
            best_score = score

        depth += 1

    except Timeout:
        return best_move

    # Return the best move from the last completed search iteration
    return best_move



  def minimax(self, grid, depth, maximizing_player=True):

    """Implement the minimax search algorithm as described in the lectures.
    """

    if time.clock() - self.start_time > self.TIMELIMIT:
      raise Timeout()

    # TODO: finish this function!

    # get moves for current grid
    moves = grid.getAvailableMoves()

    # if reached depth zero - then no need to explore any successors
    # also if reached a state where no legal moves then
    # we just return the score. "move" returned is of no value
    if depth == 0 or not moves:
      return self.score(grid), None

    # now we need to go over all the legal moves and find best
    best_score = float("-inf") if maximizing_player else float("inf")
    best_move = None

    for move in moves:
      next_game = self.forecast_move(grid, move)
      score, _ = self.minimax(next_game, depth - 1, not maximizing_player)
      if maximizing_player:
        if score >= best_score:
          best_score = score
          best_move = move
      else:
        if score <= best_score:
          best_score = score
          best_move = move
    return (best_score, best_move)

  def alphabeta(self, grid, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
    """Implement minimax search with alpha-beta pruning as described in the lectures.

    """
    if time.clock() - self.start_time > self.TIMELIMIT:
      raise Timeout()

    if depth == 0:
      return self.score(grid), None

    # now we need to go over all the legal moves and find best
    best_score = float("-inf") if maximizing_player else float("inf")
    best_move = None

    # get legal moves for current player
    moves = grid.getAvailableMoves()
    if not moves:
      return self.score(grid), None

    for move in moves:
      next_game = self.forecast_move(grid, move)
      score, _ = self.alphabeta(next_game, depth - 1, alpha, beta, not maximizing_player)
      if maximizing_player:
        if score >= best_score:
          best_score = score
          best_move = move

        if best_score >= beta:
          break

        alpha = max(alpha, best_score)

      else:
        if score <= best_score:
          best_score = score
          best_move = move

        if best_score <= alpha:
          break

        beta = min(beta, best_score)

    return best_score, best_move
