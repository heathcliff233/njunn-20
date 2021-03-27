import numpy as np
import matplotlib.pyplot as plt

src = np.loadtxt('./log.txt')
src = src.T
l1, = plt.plot(src[0])
l2, = plt.plot(src[1])
l3, = plt.plot(src[2])
plt.legend([l1,l2,l3],['D Loss', 'Tot Loss', 'Acc'])
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.show()

