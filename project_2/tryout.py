from Grid import Grid
import math

class Sample():

  def __init__(self):
    self.TIMELIMIT = 0.1*10000  # the margin to use for completing the call
    self.method = 'alphabeta' # ['minimax', 'alphabeta'] this will control what strategy is used to search
    self.start_time = None

  def display(self, grid):
    for i in xrange(grid.size):
      for j in xrange(grid.size):
        print "%6d  " % grid.map[i][j],
      print ""
    print ""

  def score(self, grid):
    smooth_wt = 0.1
    empty_wt = 2.7
    max_wt = 1.0
    monaticity_wt = 1.0

    smooth_val = self.smoothness(grid)
    empty_val = math.log(len(grid.getAvailableCells()))
    max_val = math.log(grid.getMaxTile(),2)
    monaticity_val = self.monotacity(grid)

    heuristic_val = smooth_wt*smooth_val + max_wt*max_val + empty_wt*empty_val + monaticity_wt*monaticity_val

    print 'Smooth Val={} empty_val={} max_val={} monaticity_val={}'.format(smooth_val, empty_val, max_val, monaticity_val)
    print 'HeuristicVal={}'.format(heuristic_val)
    return heuristic_val


  def smoothness(self, grid):
    #directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
    directionVectors = (DOWN_VEC, RIGHT_VEC) = ((1, 0), (0, 1))
    smooth_score = 0.0
    for x in xrange(grid.size):
      for y in xrange(grid.size):
        # if cell is occupied then find the neighbours
        if grid.map[x][y] != 0:
          for vec in directionVectors:
            for d in xrange(1, grid.size+1):
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

    # Left / Right direction
    for x in range(grid.size):
      current = 0
      next = current+1
      while next < 4:
        while next < 4 and grid.map[x][next]!= 0:
          next +=1
        if next >= 4: next -= 1
        if grid.map[x][current]!= 0:
          current_value = math.log(grid.map[x][current], 2)
        else:
          current_value = 0

        if grid.map[x][next]!= 0:
          next_value = math.log(grid.map[x][next], 2)
        else:
          next_value = 0

        if current_value > next_value:
          totals[0] += next_value - current_value
        elif next_value > current_value:
          totals[1] += current_value - next_value

        current = next
        next +=1

    # Left / Right direction
    for y in range(grid.size):
      current = 0
      next = current+1
      while next < 4:
        while next < 4 and grid.map[next][y]!= 0:
          next +=1
        if next >= 4: next -= 1
        if grid.map[current][y]!= 0:
          current_value = math.log(grid.map[current][y], 2)
        else:
          current_value = 0

        if grid.map[next][y]!= 0:
          next_value = math.log(grid.map[next][y], 2)
        else:
          next_value = 0

        if current_value > next_value:
          totals[2] += next_value - current_value
        elif next_value > current_value:
          totals[3] += current_value - next_value

        current = next
        next +=1

    return max(totals[0], totals[1]) + max(totals[2], totals[3])



if __name__ == '__main__':
    g = Grid()
    for i in [0,1,2,3]:
      for j in [0,1,2,3]:
        g.map[i][j] = 2

    g.map[2][0] = 2
    g.map[3][0] = 2

    t = Sample()
    t.display(g)
    val = t.score(g)
    print val

