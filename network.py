################################################
# --- Imports ---
################################################
import numpy as np 
from keras.datasets import imdb 
from matplotlib import pyplot 

################################################
# --- Load data ---
################################################
(X_train, y_train) , (X_test, y_test) = imdb.load_data() 
X = np.concatenate((X_train, X_test) , axis=0)
y = np.concatenate((y_train, y_test) , axis=0) 

# Summarize size 
print("Training data: ... ")
print(X.shape)
print(y.shape)
# print(X_train[0])
# Summarize number of classes
print("Classes: ")
print(np.unique(y))
# Summarize number of words
print("Number of words: ")
print(len(np.unique(np.hstack(X))))
# Summarize review length
print("Review length: ")
result = map(len, X)
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
# plot review length as a boxplot and histogram
pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()





