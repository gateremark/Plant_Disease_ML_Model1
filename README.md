# Plant Disease Classification using ResNet50

This code trains a deep learning model using the ResNet50 architecture on a dataset of plant images.
To download the Image Data set: [Kaggle Link to dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## Libraries and Modules used

* os
* tensorflow
* numpy
* matplotlib
* ImageDataGenerator
* EarlyStopping
* ModelCheckpoint

## Data Preprocessing

The training data and validation data directories are defined along with parameters like image size, batch size, shuffle size. An Image Data Generator is used to generate augmented images from the available directory images with horizontal flip, zoom range, rotation range, width shift range, height shift range, and preprocess_input (ResNet50 model-specific image preprocessing function).

```gen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    preprocessing_function=preprocess_input
)```

Next, the train_generator and val_generator are defined using flow_from_directory() function pointing to the respective train and test directories.

```train_generator = gen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_size, img_size)
)

val_generator = gen.flow_from_directory(
    directory=val_data_dir,
    target_size=(img_size, img_size)
)```


## Custom Metric Functions

Custom metrics functions like recall, precision, and F1 score are defined using Keras backend (K) function for evaluating the performance of the model later. 

```from tensorflow.keras import backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_m = true_positives / (possible_positives + K.epsilon())
    return recall_m

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_m = true_positives / (predicted_positives + K.epsilon())
    return precision_m

def f1(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))```

## Model Architecture and Compilation

The ResNet50 architecture is imported as a pre-trained network (trained on the ImageNet dataset) with its top layer removed. A new fully connected layer is attached to the previous layer to set up a pipeline to classify 39 plant diseases. The final model is compiled using the Adam optimizer function and categorical cross-entropy loss.

```resnet50 = ResNet50(include_top=False, weights="imagenet", input_shape=img_shape)
resnet50.trainable = False

x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(39, activation="softmax")(x)

model = Model(resnet50.input, x)

model.compile(optimizer=Adam(learning_rate=0.00015), loss='categorical_crossentropy', metrics=['acc', f1, precision, recall])```

Several callbacks are used during the training process to monitor and save the best model.

```checkpoint = ModelCheckpoint("resnet50_v1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')```

## Model Training and Evaluation

The model is then trained using fit() function over 20 epochs.

```model_history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=20,
    validation_data=val_generator,
    validation_steps=25,
    callbacks = [checkpoint, early]
)```

After the training, the accuracy, loss, f1-score, precision and recall metrics are plotted using matplotlib.

```plt.plot(model_history.history['loss'], label = 'training loss')
plt.plot(model_history.history['acc'], label = 'training accuracy')
plt.grid(True)
plt.legend()

plt.plot(model_history.history['val_loss'], label = 'validation loss')
plt.plot(model_history.history['val_acc'], label = 'validation accuracy')
plt.grid(True)
plt.legend()```

## Prediction on Test Images

Model predictions can be made on test images by calling the `get_prediction` function. In this case, it predicts the disease type of a tomato image.

```def preprocess_image(file):
  path = '/content/drive/MyDrive/digital-farmer/tests/'
  img = image.load_img(path + file, target_size=(256,256)) #return (img_size, img_size
  img_array = image.img_to_array(img)
  img_array_expanded_dims = np.expand_dims(img_array, axis=0)
  return tf.keras.applications.resnet.preprocess_input(img_array_expanded_dims)
  # replace mobilenet with resnet

def get_prediction(image, level='all'):
  preprocessed_image = preprocess_image(image)
  predictions = model.predict(preprocessed_image)

  ind = np.argpartition(predictions[0], -5)[-5:]
  result = np.argmax(predictions[0])
  top5= predictions[0][ind]

  if level == 'single':
     for k,v in labels.items():
       if v == result: return k;
  else:
    for k,v in labels.items():
      if v in np.sort(ind):
        idx = np.where(ind == v)[0]
        print(f'{top5[idx]} ~> {k}')

from IPython.display import Image
Image(filename='/content/drive/MyDrive/digital-farmer/tests/tomato_bacterial-spot.jpg', width=300, height=200)

ans = get_prediction('tomato_bacterial-spot.jpg')
ans```

The above code will output the predicted disease type of the given image.
