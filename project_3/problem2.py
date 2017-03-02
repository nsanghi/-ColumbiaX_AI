import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

  X = []
  y = []
  with open(sys.argv[1], 'rb') as f1:
    with open (sys.argv[2], 'wb') as f2:
      inputreader = csv.reader(f1)
      outputwriter = csv.writer(f2)
      for row in inputreader:
        X.append([1.0, float(row[0]), float(row[1])])
        y.append(float(row[2]))

      X = np.array(X)
      y = np.array(y)

      # noramilise two features
      for i in [1,2]:
        m = np.mean(X[:,i])
        sd = np.std(X[:,i])
        X[:,i] = (X[:,i]-m)/sd


      n = X.shape[0]
      alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.2]
      iterations = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
      convergence = np.zeros((100, 10))

      for i in range(len(alphas)):
        beta = [0.0, 0.0, 0.0]
        alpha = alphas[i]
        max_iter = iterations[i]

        for p in range(max_iter):
          pred = np.dot(X, beta)
          error = pred-y
          mse = np.dot(error, error) / float(2*n)
          #copy error
          convergence[p,i] = mse

          beta_n = np.copy(beta)
          for j in range(3):
            beta_n[j] = beta[j] - alpha/float(n) * np.sum(error * X[:, j])
          beta = np.copy(beta_n)

        # write final beta for each alpha
        print beta
        print np.concatenate(([alpha, max_iter], beta))
        outputwriter.writerow(np.concatenate(([alpha, max_iter], beta)))

  f1.close()
  f2.close()

  num_plots = len(alphas)
  colormap = plt.cm.gist_ncar
  plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
  plt.ylim((0, 1.5))

  labels = []
  for i in range(num_plots):
    plt.plot(np.linspace(0,99,100) , convergence[:, i])
    labels.append('alpha={}'.format(alphas[i]))

  plt.legend(labels, ncol = 5, loc = 'upper center', bbox_to_anchor = [0.5, 1.1],
              columnspacing = 1.0, labelspacing = 0.0, handletextpad = 0.0,
              handlelength = 1.5, fancybox = True, shadow = True)

  plt.show()

