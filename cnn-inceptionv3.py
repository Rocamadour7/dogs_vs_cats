'''
Author: Luis Molina
Date: July 5th, 2017
Based on the code from Abnera in his image-classifier
Link: https://github.com/abnera/image-classifier
Data: https://www.kaggle.com/c/dogs-vs-cats/data
Data should be put in categorial folders for keras ImageDataGenerator Class
Directory Structure:
    Data/
    |--Train/
    |------cats/
    |------dogs/
    |--Validation/
    |------cats/
    |------dogs/
Test and sample testing doesn't need manual preprocess since they aren't label and just used to predict
Model took around 6 hours to train on a GTX 1060 6GB, it could take longer if it keeps improving
'''

import os
# Keras imports
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
# Numpy for array operations
import numpy as np
# PLT just to plot my own test pictures
import matplotlib.pyplot as plt
# Visual feedback for the For loops.
from tqdm import tqdm

# Directories
# Dataset could be obtained from Kaggle
# Adjust to were you are saving the dataset
# The sample path are my own test pictures
from tqdm._tqdm import tqdm

TRAIN_DIR = 'D:/Code/Deep Learning A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/dataset/training_set'
VALIDATION_DIR = 'D:/Code/Deep Learning A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/dataset/test_set'
MODEL_PATH = 'D:/Code/Deep Learning A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/model/inceptionV3'
SAMPLE_PATH = 'D:/Code/Deep Learning A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/dataset/single_prediction'
SUBMISSION_PATH = 'D:/Code/Kaggle_Data/Dogs_vs_Cats/test'

# Parameters
nb_classes = 2
img_width, img_height = 299, 299
batch_size = 32
nb_epoch = 50
learning_rate = 1e-4
momentum = 0.9
transformation_ratio = 0.05


# Function to train top layer of inception
def train_inception():
    # Call InceptionV3, input shape must be 299 by 299, if you need to change that you will need to use the input method
    base_model = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    # Own little cnn on top of the model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # Build the model with inceptionV3 and our own cnn
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the inception layers, since they don't need to be retrained
    for layer in base_model.layers:
        layer.trainable = False

    # Create a class for the images
    # I use the preprocessing_function of the inceptionV3 class, not sure if needed, almost same results with rescale=1/255.
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=transformation_ratio,
                                       shear_range=transformation_ratio,
                                       zoom_range=transformation_ratio,
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Method for flowing the images from a directory
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    # Compile our model with RMSProp
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Directory to save the weights of the top model
    top_weights_path = os.path.join(os.path.abspath(MODEL_PATH), 'top_model_weights.h5')
    # Defining the callback list for checkpoints and to be saving the best epochs
    callback_list = [
            ModelCheckpoint(top_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=5, verbose=0)
            ]

    # Train Simple CNN
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples//batch_size,
                        epochs=nb_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples//batch_size,
                        callbacks=callback_list)
    
    print('\nStarting to Fine Tune Model\n')

    # We load the best epoch weights from the top model
    model.load_weights(top_weights_path)

    # Just to show all the layers
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # Train the very last layers and the top model
    for layer in base_model.layers[:249]:
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True

    # SGD optimizer, slower and memory heavier, but used to prevent local minima.
    optimizer = SGD(lr=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Directory and callbacks for the Fine-Tuning training
    final_weights_path = os.path.join(os.path.abspath(MODEL_PATH), 'model_weights.h5')
    callback_list = [
            ModelCheckpoint(final_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=5, verbose=0)
            ]

    # Fine-Tune the model
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples//batch_size,
                        epochs=nb_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples//batch_size,
                        callbacks=callback_list)

    # Save the model into a JSON file
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(MODEL_PATH), 'model.json'), 'w') as json_file:
        json_file.write(model_json)
        json_file.close()
    return model
    # val_loss = 0.03310 and val_acc = 0.9869


# Function to make predictions on my own photos, 4x4 plot
def make_prediction(model):
    # Empty dictionary to fill with the name of the file, label and the result %
    testing_data = []
    fig = plt.figure()
    for num, img in tqdm(enumerate(os.listdir(SAMPLE_PATH))):
        # Define the path of the img
        img_path = os.path.join(SAMPLE_PATH, img)
        # Save the name on the variable without the extension
        file_name = img.split('.')[0]
        # Load img
        img = image.load_img(img_path, target_size=(img_height, img_width))
        # Convert img to array
        x = image.img_to_array(img)
        # Rescale img to show for matplotlib
        orig = x/255.
        # Preprocess the input, again this may not be really needed and only a x/255. would be enough
        x = preprocess_input(x)
        # Reshape the img to (batch_size, img_width, img_height, channels)
        x = np.expand_dims(x, axis=0)
        # Call our model to predict [cat-likeness dog-likeness]
        result = model.predict(x)
        # Label the prediction
        if np.argmax(result) == 1:
            label = 'Dog'
        elif np.argmax(result) == 0:
            label = 'Cat'
        # Add the subplot to the figure defined on top 4x4 subplot
        y = fig.add_subplot(4, 4, num + 1)
        # Add img to the subplot
        y.imshow(orig)
        # Add title with the label
        plt.title(label)
        # Hide the axis
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        # Append the data to our dict.
        testing_data.append([file_name, label, result[0][np.argmax(result)]])
    # Show the plot with all img (16 for me)
    plt.show()
    # Return our dict
    return testing_data


# Function to generate and csv file to submit it to Kaggle
def make_submission_file(model):
    # Open and create the file and add the headers
    with open(os.path.join(os.path.abspath(MODEL_PATH), 'submission_file.csv'), 'w') as f:
        f.write('id,label\n')
        f.close()

    # Fill each line with the id and prediction of being a dog
    with open(os.path.join(os.path.abspath(MODEL_PATH), 'submission_file.csv'), 'a') as f:
        for img in tqdm(os.listdir(SUBMISSION_PATH)):
            img_path = os.path.join(SUBMISSION_PATH, img)
            img_num = img.split('.')[0]
            img = image.load_img(img_path, target_size=(img_height, img_width))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            result = model.predict(x)
            result = result[0][1]
            f.write('{},{}\n'.format(img_num, result))
        f.close()


def main():
    # Checking to see if a model exists to just load it, if not train the model.
    if not (not os.path.exists(os.path.join(os.path.abspath(MODEL_PATH), 'model.json')) or not os.path.exists(
            os.path.join(os.path.abspath(MODEL_PATH), 'model_weights.h5'))):
        with open(os.path.join(os.path.abspath(MODEL_PATH), 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(os.path.abspath(MODEL_PATH), 'model_weights.h5'))
        print('Model loaded!')
    else:
        print('Training...this may take a while')
        model = train_inception()
    predictions = make_prediction(model)
    print(predictions)
    make_submission_file(model)


if __name__ == '__main__':
    main()






















