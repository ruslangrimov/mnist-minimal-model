# The minimal neural network that achieves 99% on MNIST

Here I am trying to find out what is the **minimal** model in terms of number of parameters that can achieve **99% accuracy** on **MNIST** dataset.

All sub folders in this repository contains model.txt file with description of a model and hdf5 files with its weights. The subfolder name is the number of trainable parameters of the model.

The record holder at the moment is the next network:

```python
def dw_block(sh_l, prev_l):
    l = sh_l(prev_l)
    l = Activation(activation='relu')(l)
    l = InstanceNormalization(axis=None)(l)
    l = Dropout(rate=0.1)(l)
    return l
    
inputs = Input((28, 28, 1), dtype=np.float32)
l = inputs

l = Conv2D(8, (3, 3), input_shape=input_shape)(l)
l = Activation(activation='relu')(l)
l = InstanceNormalization(axis=None)(l)
l = MaxPooling2D((2, 2))(l)
l = Dropout(rate=0.1)(l)

l = SeparableConv2D(26, (3, 3), depth_multiplier=1)(l)
l = Activation(activation='relu')(l)
l = InstanceNormalization(axis=None)(l)
l = Dropout(rate=0.1)(l)

sh_l = SeparableConv2D(26, (3, 3), depth_multiplier=1, padding='same')

for n in range(3):
    l = dw_block(sh_l, l)

l = GlobalAveragePooling2D()(l)

l = Dense(16, activation='relu')(l)
l = InstanceNormalization(axis=None)(l)
l = Dropout(rate=0.1)(l)
l = Dense(10, activation='softmax')(l)

model = Model(inputs, l)

batch_size = 32
num_epochs = 75

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

w_path = os.path.join('mini', model.name)

if not os.path.exists(w_path):
    os.makedirs(w_path)

f_name = os.path.join(w_path, 'weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5')

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1,
          callbacks=[ModelCheckpoint(f_name)])

```

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 28, 28, 1)    0                                            
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 26, 26, 8)    80          input_2[0][0]                    
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 26, 26, 8)    0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
instance_normalization_1 (Insta (None, 26, 26, 8)    2           activation_7[0][0]               
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 13, 13, 8)    0           instance_normalization_1[0][0]   
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 13, 13, 8)    0           max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
separable_conv2d_3 (SeparableCo (None, 11, 11, 26)   306         dropout_8[0][0]                  
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 11, 11, 26)   0           separable_conv2d_3[0][0]         
__________________________________________________________________________________________________
instance_normalization_2 (Insta (None, 11, 11, 26)   2           activation_8[0][0]               
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 11, 11, 26)   0           instance_normalization_2[0][0]   
__________________________________________________________________________________________________
separable_conv2d_4 (SeparableCo (None, 11, 11, 26)   936         dropout_9[0][0]                  
                                                                 dropout_10[0][0]                 
                                                                 dropout_11[0][0]                 
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 11, 11, 26)   0           separable_conv2d_4[0][0]         
__________________________________________________________________________________________________
instance_normalization_3 (Insta (None, 11, 11, 26)   2           activation_9[0][0]               
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 11, 11, 26)   0           instance_normalization_3[0][0]   
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 11, 11, 26)   0           separable_conv2d_4[1][0]         
__________________________________________________________________________________________________
instance_normalization_4 (Insta (None, 11, 11, 26)   2           activation_10[0][0]              
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, 11, 11, 26)   0           instance_normalization_4[0][0]   
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 11, 11, 26)   0           separable_conv2d_4[2][0]         
__________________________________________________________________________________________________
instance_normalization_5 (Insta (None, 11, 11, 26)   2           activation_11[0][0]              
__________________________________________________________________________________________________
dropout_12 (Dropout)            (None, 11, 11, 26)   0           instance_normalization_5[0][0]   
__________________________________________________________________________________________________
global_average_pooling2d_2 (Glo (None, 26)           0           dropout_12[0][0]                 
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 16)           432         global_average_pooling2d_2[0][0] 
__________________________________________________________________________________________________
instance_normalization_6 (Insta (None, 16)           2           dense_3[0][0]                    
__________________________________________________________________________________________________
dropout_13 (Dropout)            (None, 16)           0           instance_normalization_6[0][0]   
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 10)           170         dropout_13[0][0]                 
==================================================================================================
Total params: 1,936
Trainable params: 1,936
Non-trainable params: 0

```

```
Test score:  0.02555989224165678
Test accuracy:  0.991
```
