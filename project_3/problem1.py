import csv
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':

  X = []
  y = []
  with open(sys.argv[1], 'rb') as f1:
    with open (sys.argv[2], 'wb') as f2:
      inputreader = csv.reader(f1)
      outputwriter = csv.writer(f2)
      for row in inputreader:
        X.append([float(row[0]), float(row[1]), 1.0])
        y.append(float(row[2]))

      w = [0.0,0.0,0.0]
      flag = True
      while flag:
        old_w = list(w)
        for i in range(len(X)):
          xi = X[i]
          yi = y[i]
          predictor = sum( [xi[d]*w[d] for d in range(3)] )

          if predictor*yi <= 0:
            w = [w[d]+yi*xi[d] for d in range(3)]

        outputwriter.writerow(w)

        if old_w == w:
          flag = False

        # plot each iteration
        x1 = [x[0] for x in X]
        x2 = [x[1] for x in X]
        plt.scatter(x1, x2, c=y)
        x1 = [i for i in range(15)]
        x2 = [(-w[2] - w[0] * x1[i]) / w[1] for i in range(15)]
        plt.plot(x1, x2)
        plt.show()

  f1.close()
  f2.close()

