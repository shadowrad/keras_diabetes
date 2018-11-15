# Create your first MLP in Keras
import os
import random

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset

local_path = os.getcwd()
print(local_path)

dataset = numpy.loadtxt("C:\\Users\\rodrigo\\PycharmProjects\\keras_test\\data\\diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

prueba_random=random.randint(0,len(dataset)-1)

test_x = X[prueba_random-1:prueba_random]
test_y = Y[prueba_random-1:prueba_random]

prediction = model.predict(test_x )

# round predictions
rounded = [round(x[0]) for x in prediction]
print("predigo que es:"+str(rounded))
print("En el caso es:"+str(test_y))

