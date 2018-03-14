# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.preprocessing.image import *

np.random.seed(2018)


def link(basedir, file_name, driver_imgs_list, driver_id_list):
    os.mkdir(basedir + file_name)
    for i in range(10):
        os.mkdir(basedir + file_name + "/c%d"%i)
    for driver_id in driver_id_list:
        df_part = driver_imgs_list[driver_imgs_list['subject'] == driver_id]
        for index, row in df_part.iterrows():
            subpath = row["classname"] + "/" + row["img"]
            src = basedir + "train/" + subpath 
            dst = basedir + file_name + "/" + subpath
            os.link(src, dst)


def train_valid_split(basedir, group, driver_imgs_list, driver_list, valid_drivers):
    train_drivers = [i for i in driver_list if i not in valid_drivers]
    link(basedir, 'train'+str(group), driver_imgs_list, train_drivers)
    link(basedir, 'valid'+str(group), driver_imgs_list, valid_drivers)

    
def test_gen(gen, basedir, model_image_size, batch_size):
    
    test_generator = gen.flow_from_directory(os.path.join(basedir, 'Test'), target_size=model_image_size, shuffle=False, batch_size=batch_size, class_mode=None)
    
    steps_test_sample = test_generator.samples//batch_size + 1
    return test_generator, steps_test_sample

    
def tv_gen(group, train_gen, valid_gen, basedir, model_image_size, batch_size):
    
    train_generator = train_gen.flow_from_directory(os.path.join(basedir, 'train'+str(group)),  model_image_size, shuffle=True, batch_size=batch_size, class_mode="categorical")

    valid_generator = valid_gen.flow_from_directory(os.path.join(basedir, 'valid'+str(group)),  model_image_size, shuffle=True, batch_size=batch_size, class_mode="categorical")
    
    steps_train_sample = int(train_generator.samples)//batch_size + 1
    steps_valid_sample = int(valid_generator.samples)//batch_size + 1
    return train_generator, valid_generator, steps_train_sample, steps_valid_sample


def build_model(model_name, model_image_size, fune_tune_layer):
    inputs = Input((*model_image_size, 3))
    base_model = model_name(input_tensor=inputs, weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs, x)    
    for i in range(fune_tune_layer):
        model.layers[i].trainable = False
    return model


def train_model(model, optimizer, epoch, train_generator, valid_generator, steps_train_sample, steps_valid_sample):
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=epoch, validation_data=valid_generator, validation_steps=steps_valid_sample)
    
    
def predict_model(model, test_generator, steps_test_sample):
    y_pred = model.predict_generator(test_generator, steps=steps_test_sample, verbose=1)
    return y_pred


def submission(df, result, fname):
    for i in tqdm(range(result.shape[0])):
        df.iloc[i,1:11] = result[i]    
    df.to_csv(fname, index=None)