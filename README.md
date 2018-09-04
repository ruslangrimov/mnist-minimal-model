# The minimal neural network that achieves 99% on MNIST

Here I am trying to find out what is the **minimal** model in terms of number of parameters that can achieve **99% accuracy** on **MNIST** dataset.

All sub folders in this repository contains model.txt file with description of a model and hdf5 files with its weights. The subfolder name is the number of trainable parameters of the model.

The record holder at the moment is the CNN with 2,720 parameters, 3x3 convolution, separable convolutions, maxpooling, batchnormalization and dropouts:

```python
def dw_block(sh_l, prev_l):
    l = sh_l(prev_l)
    l = Activation(activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(rate=0.1)(l)
    return l
    
inputs = Input((28, 28, 1), dtype=np.float32)
l = inputs

l = Conv2D(8, (3, 3), input_shape=input_shape)(l)
l = Activation(activation='relu')(l)
l = BatchNormalization()(l)
l = MaxPooling2D((2, 2))(l)
l = Dropout(rate=0.1)(l)

l = SeparableConv2D(26, (3, 3), depth_multiplier=1)(l)
l = Activation(activation='relu')(l)
l = BatchNormalization()(l)
l = Dropout(rate=0.1)(l)

sh_l = SeparableConv2D(26, (3, 3), depth_multiplier=1, padding='same')

for n in range(3):
    l = dw_block(sh_l, l)

l = GlobalAveragePooling2D()(l)

l = Dense(16, activation='relu')(l)
l = BatchNormalization()(l)
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
input_16 (InputLayer)           (None, 28, 28, 1)    0                                            
__________________________________________________________________________________________________
conv2d_80 (Conv2D)              (None, 26, 26, 8)    80          input_16[0][0]                   
__________________________________________________________________________________________________
activation_386 (Activation)     (None, 26, 26, 8)    0           conv2d_80[0][0]                  
__________________________________________________________________________________________________
batch_normalization_419 (BatchN (None, 26, 26, 8)    32          activation_386[0][0]             
__________________________________________________________________________________________________
max_pooling2d_80 (MaxPooling2D) (None, 13, 13, 8)    0           batch_normalization_419[0][0]    
__________________________________________________________________________________________________
dropout_402 (Dropout)           (None, 13, 13, 8)    0           max_pooling2d_80[0][0]           
__________________________________________________________________________________________________
separable_conv2d_79 (SeparableC (None, 11, 11, 26)   306         dropout_402[0][0]                
__________________________________________________________________________________________________
activation_387 (Activation)     (None, 11, 11, 26)   0           separable_conv2d_79[0][0]        
__________________________________________________________________________________________________
batch_normalization_420 (BatchN (None, 11, 11, 26)   104         activation_387[0][0]             
__________________________________________________________________________________________________
dropout_403 (Dropout)           (None, 11, 11, 26)   0           batch_normalization_420[0][0]    
__________________________________________________________________________________________________
separable_conv2d_80 (SeparableC (None, 11, 11, 26)   936         dropout_403[0][0]                
                                                                 dropout_404[0][0]                
                                                                 dropout_405[0][0]                
__________________________________________________________________________________________________
activation_388 (Activation)     (None, 11, 11, 26)   0           separable_conv2d_80[0][0]        
__________________________________________________________________________________________________
batch_normalization_421 (BatchN (None, 11, 11, 26)   104         activation_388[0][0]             
__________________________________________________________________________________________________
dropout_404 (Dropout)           (None, 11, 11, 26)   0           batch_normalization_421[0][0]    
__________________________________________________________________________________________________
activation_389 (Activation)     (None, 11, 11, 26)   0           separable_conv2d_80[1][0]        
__________________________________________________________________________________________________
batch_normalization_422 (BatchN (None, 11, 11, 26)   104         activation_389[0][0]             
__________________________________________________________________________________________________
dropout_405 (Dropout)           (None, 11, 11, 26)   0           batch_normalization_422[0][0]    
__________________________________________________________________________________________________
activation_390 (Activation)     (None, 11, 11, 26)   0           separable_conv2d_80[2][0]        
__________________________________________________________________________________________________
batch_normalization_423 (BatchN (None, 11, 11, 26)   104         activation_390[0][0]             
__________________________________________________________________________________________________
dropout_406 (Dropout)           (None, 11, 11, 26)   0           batch_normalization_423[0][0]    
__________________________________________________________________________________________________
global_average_pooling2d_37 (Gl (None, 26)           0           dropout_406[0][0]                
__________________________________________________________________________________________________
dense_105 (Dense)               (None, 16)           432         global_average_pooling2d_37[0][0]
__________________________________________________________________________________________________
batch_normalization_424 (BatchN (None, 16)           64          dense_105[0][0]                  
__________________________________________________________________________________________________
dropout_407 (Dropout)           (None, 16)           0           batch_normalization_424[0][0]    
__________________________________________________________________________________________________
dense_106 (Dense)               (None, 10)           170         dropout_407[0][0]                
==================================================================================================
Total params: 2,436
Trainable params: 2,180
Non-trainable params: 256

```

```
Test score:  0.027863642352301394
Test accuracy:  0.9915
```
