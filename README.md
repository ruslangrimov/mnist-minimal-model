# The minimal neural network that achieves 99% on MNIST

Here I am trying to find out what is the **minimal** model in terms of number of parameters that can achieve **99% accuracy** on **MNIST** dataset.

All sub folders in this repository contains model.txt file with description of a model and hdf5 files with its weights. The subfolder name is the number of trainable parameters of the model.

The record holder at the moment is the CNN with 3,295 parameters, 3x1+1x3 and 1x1 convolutions and dropouts:

```python
model.add(Conv2D(8, (3, 3), input_shape=input_shape))
model.add(Activation(activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.1))

model.add(Conv2D(12, (3, 1)))
model.add(Activation(activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(12, (1, 3)))
model.add(Activation(activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.1))

model.add(Conv2D(15, (3, 1)))
model.add(Activation(activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(15, (1, 3)))
model.add(Activation(activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.1))

model.add(Conv2D(4, (1, 1)))
model.add(Activation(activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.1))

model.add(Flatten())

model.add(Dense(20, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.1))
model.add(Dense(10, activation='softmax'))

batch_size = 32
num_epochs = 75

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1,
          callbacks=[ModelCheckpoint('mini/weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5',
                                     monitor='loss,acc,val_loss,val_acc')])


```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_25 (Conv2D)           (None, 26, 26, 8)         80        
_________________________________________________________________
activation_25 (Activation)   (None, 26, 26, 8)         0         
_________________________________________________________________
batch_normalization_36 (Batc (None, 26, 26, 8)         32        
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 13, 13, 8)         0         
_________________________________________________________________
batch_normalization_37 (Batc (None, 13, 13, 8)         32        
_________________________________________________________________
dropout_21 (Dropout)         (None, 13, 13, 8)         0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 11, 13, 12)        300       
_________________________________________________________________
activation_26 (Activation)   (None, 11, 13, 12)        0         
_________________________________________________________________
batch_normalization_38 (Batc (None, 11, 13, 12)        48        
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 11, 11, 12)        444       
_________________________________________________________________
activation_27 (Activation)   (None, 11, 11, 12)        0         
_________________________________________________________________
batch_normalization_39 (Batc (None, 11, 11, 12)        48        
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 5, 5, 12)          0         
_________________________________________________________________
batch_normalization_40 (Batc (None, 5, 5, 12)          48        
_________________________________________________________________
dropout_22 (Dropout)         (None, 5, 5, 12)          0         
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 3, 5, 15)          555       
_________________________________________________________________
activation_28 (Activation)   (None, 3, 5, 15)          0         
_________________________________________________________________
batch_normalization_41 (Batc (None, 3, 5, 15)          60        
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 3, 3, 15)          690       
_________________________________________________________________
activation_29 (Activation)   (None, 3, 3, 15)          0         
_________________________________________________________________
batch_normalization_42 (Batc (None, 3, 3, 15)          60        
_________________________________________________________________
dropout_23 (Dropout)         (None, 3, 3, 15)          0         
_________________________________________________________________
conv2d_30 (Conv2D)           (None, 3, 3, 4)           64        
_________________________________________________________________
activation_30 (Activation)   (None, 3, 3, 4)           0         
_________________________________________________________________
batch_normalization_43 (Batc (None, 3, 3, 4)           16        
_________________________________________________________________
dropout_24 (Dropout)         (None, 3, 3, 4)           0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 36)                0         
_________________________________________________________________
dense_9 (Dense)              (None, 20)                740       
_________________________________________________________________
batch_normalization_44 (Batc (None, 20)                80        
_________________________________________________________________
dropout_25 (Dropout)         (None, 20)                0         
_________________________________________________________________
dense_10 (Dense)             (None, 10)                210       
=================================================================
Total params: 3,507
Trainable params: 3,295
Non-trainable params: 212

```

```
Test score:  0.0271674046346
Test accuracy:  0.9912
```
