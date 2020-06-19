import tensorflow as tf 
import pandas as pd
import numpy as np 
import cv2

class Recorder:

    def __init__(self,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                valid_indices,
                resize_shape,
                test_df = None):
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.valid_images = valid_images
        self.valid_labels = valid_labels
        self.valid_indices = valid_indices
        self.resize_shape = resize_shape
        self.test_df = test_df 

    def byte_feature(self,value):
        """ Returns a bytes_list from string / byte """ 
        if isinstance(value,type(tf.constant(0))):
            value = value.numpy() 
        return tf.train.Feature(bytes_list= tf.train.BytesList(value=[value]))
    
    def float_feature(self,value):
        """ Returns a float_list from a float / double """ 
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def int_feature(self,value):
        """ Returns a int64_list from a bool / int / list """ 
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_train_example(self,feature0,feature1):

        feature = {
            'image': self.byte_feature(feature0),
            'target': self.int_feature(feature1)
        } 

        example = tf.train.Example(features = tf.train.Features(feature=feature))

        return example.SerializeToString()
    
    def serialize_val_example(self,feature0,feature1,feature2):

        feature = {
            'image': self.byte_feature(feature0),
            'target': self.int_feature(feature1),
            'index': self.int_feature(feature2)
        }

        example = tf.train.Example(features= tf.train.Features(feature=feature))

        return example.SerializeToString()

    def Writer(self,fold_num):

        with tf.io.TFRecordWriter(f'train_fold_{fold_num}.tfrec') as writer:

            for image,label in zip(self.train_images,self.train_labels):

                img = cv2.imread(image)
                img = cv2.resize(img,self.resize_shape,cv2.INTER_AREA)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                img = cv2.imencode('.jpg',img,(cv2.IMWRITE_JPEG_QUALITY,100))[1].tostring()

                example = self.serialize_train_example(img,label)

                writer.write(example)
                print(f"Saved image {image} with label {label}")
            
        
        print("Started to write validation examples")

        with tf.io.TFRecordWriter(f"validation_fold_{fold_num}.tfrec") as writer:

            for image,label,index in zip(self.valid_images,self.valid_labels,self.valid_indices):

                img = cv2.imread(image)
                img = cv2.resize(img,self.resize_shape,cv2.INTER_AREA)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                img = cv2.imencode('.jpg',img,(cv2.IMWRITE_JPEG_QUALITY,100))[1].tostring()

                example = self.serialize_val_example(img,label,index)

                writer.write(example)

                print(f"Saved image {image} with label {label} and index {index}")