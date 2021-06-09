## Task 2: Import Libraries and Select the Module
# Download the classifier Select the TF2 SavedModel module to use

import itertools
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
# import onnxmltools

from azureml.core import Run

run = Run.get_context()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='.',
        help='Path to the training data'
    )

    args = parser.parse_args()

    # Download the classifier Select the TF2 SavedModel module to use
    module_selection = ("mobilenet_v2_100_224", 224) 
    handle_base, pixels = module_selection
    MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
    IMAGE_SIZE = (pixels, pixels)
    print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

    BATCH_SIZE = 32 

    ## Task 3: Setup the dataset with Augmentation
    # Inputs are suitably resized for the selected module 

    data_dir = args.data_path

    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                            interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    do_data_augmentation = True
    if do_data_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2,
            **datagen_kwargs
        )
    else:
        train_datagen = valid_datagen

    train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=True, **dataflow_kwargs
    )


    # Download the classifier Select the TF2 SavedModel module to use

    do_fine_tuning = False 

    print("Building model with", MODULE_HANDLE)
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
        hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(train_generator.num_classes,
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+IMAGE_SIZE+(3,))
    model.summary()

    ## Task 4: Train the model and Visualize Results

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy']
    )

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    hist = model.fit(
        train_generator,
        epochs=15, steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps).history

    # plt.figure()
    # plt.ylabel("loss (training and validation")
    # plt.xlabel("training Steps")
    # plt.ylim([0,2])
    # plt.plot(hist["loss"])
    # plt.plot(hist["val_loss"])

    # plt.figure()
    # plt.ylabel("Accuracy (training and validation)")
    # plt.xlabel("training Steps")
    # plt.ylim([0,1])
    # plt.plot(hist["accuracy"])
    # plt.plot(hist["val_accuracy"])

    ## Task 5: Test the model on Validation Data

    def get_class_string_from_index(index):
        for class_string, class_index in valid_generator.class_indices.items():
            if class_index == index:
                return class_string

    x, y = next(valid_generator)
    image = x[0, :, :, :]
    true_index = np.argmax(y[0])
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    # Expand the validation image to (1, 224, 224, 3) before predicting the label
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("True label: " + get_class_string_from_index(true_index))
    print("Predicted label: " + get_class_string_from_index(predicted_index))

 

    # saved_model_path = "../outputs"
    # tf.saved_model.save(model, saved_model_path)

    model.save('../outputs/mic_model.h5')

    # tf.keras.experimental.export_saved_model(model, 'mic_mol.h5')

    #Registro
    # with open('./outputs/model.pkl', 'wb') as model_pkl:
    #     pickle.dump(model, model_pkl)


    # model.AvgPool2d(input_size - (output_size - 1) * (input_size // output_size), stride=input_size // output_size)
    # onnx_model = onnxmltools.convert_keras(model) 

    # onnxmltools.utils.save_model(onnx_model, '../outputs/mic-model.onnx')
    