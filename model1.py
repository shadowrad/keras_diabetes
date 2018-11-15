from keras.layers import Input, Dense
from keras.models import Model


data = [[140, 1], [130, 1], [150, 0], [170, 0],[180, 1], [120, 1], [142, 0], [150, 0]]
labels = [0, 1, 1, 1, 1, 1, 1, 0]

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training