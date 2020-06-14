import tensorflow as tf 
import tensorflow_addons as tfa
import math 
import numpy as np 
import random
import os 
class Dataset:

    def __init__(self,
                train_files,
                test_files,
                validation_files,
                validation_split,
                image_size,
                dataset_size,
                batch_size,
                shuffle,
                resize_shape = None):
        
        """
           Args:
           - train_files - tfrecord file that contains training data 
           NOTE: remember to use glob to get all the files from the train directory.
           - test_files - tfrecord file that contains test data 
           - validation_files - tfrecord file that contains validation data
           - validation_split - percentage of training data that is split into validation data
           - image_size - image_size in tfrecord
           - dataset_size - number of examples in trainingset
           - batch_size - height of batch_size
           - shuffle - int describint amount of data to be shuffled
           - resize_shape - shape to which you want to resize your images - tuple [width,height]
        """
        
        self.train_files = train_files
        self.test_files = test_files
        self.validation_files = validation_files
        self.validation_split = validation_split 
        self.image_size = image_size
        self.dataset_size = dataset_size
        self.batch_size = batch_size 
        self.resize_shape = resize_shape

    
    def seed_everything(self,seed):
        """
            Place a seed for the random number generator
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    def decode_image(self,
                        img):
        """
            This function decode JPEG image, because it was saved as a byte (or string) feature.
            Args:
             - img - image in a string format
            Returns:
             - Image casted as a tf.float32 tensor object with size [1024,1024,3]
        """
        image = tf.image.decode_jpeg(img,channels=3)
        image = tf.cast(image,tf.float32) / 255.0
        image = tf.reshape(image,[*self.image_size,3])
        return image

    def decode_image_from_raw_jpeg(self,
                                    filename,
                                    label):
        """
            Decode image from raw jpeg filename and parse it to format comatible with tensorflow
            Args:
            filename - raw jpeg file path.
            label - If the image has corresponding label.
            Returns: 
            If label is set to None, function returns only image,
            when set to True, returns image with corresponding label
        """
        bits = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(bits,channels=3)
        image = tf.cast(image,tf.float32) / 255.0
        image = tf.image.resize(image,[*self.resize_shape])

        return image,label
    
    def read_labeled_tfrecord(self,
                                example):
        """
           In this function you read images nad labels from tfrecord in their
           default data type and parse them to a format where they can be 
           fed into model.
           Args:
           - example - tfrecord label contains image and corresponding label
           Returns:
           - Image - image casted as a tf.float32 tensor object with size [1024,1024,3]
           - label - image corresponding label in tf.int32 format 
        """
        feature = {
            "image": tf.io.FixedLenFeature([],tf.string),
            "target": tf.io.FixedLenFeature([],tf.int64)
        }
        example = tf.io.parse_single_example(example,feature)
        image = self.decode_image(example['image'])
        label = tf.cast(example['target'],tf.int32)

        return image,label
    
    def read_unlabeled_tfrecord(self,
                                example):
        """
            This function is similiar to a read_labeled_tfrecord function 
            but here instead of having label we have image_name.
            Args:
            - example - tfrecord label contains image and coressponding label
            Returns:
            - Image - image casted as a tf.float32 tensor object with size [1024,1024,3]
            - idnum - image name corresponding to parsed image
        """
        feature = {
            "image": tf.io.FixedLenFeature([],tf.string),
            "image_name": tf.io.FixedLenFeature([],tf.string)
        }
        example = tf.io.parse_single_example(example,feature)
        image = self.decode_image(example['image'])
        idnum = example['image_name']

        return image,idnum

    def data_augment_and_resize(self,
                    image,
                    label):
        """
            Simple Data Augmentation that is compatible with tensorflow 
            and also added resize to compress the image a little bit 
            so that It will consume less memory.
            Remember to use this function in map method provided by tensorflow
            Args:
            - image - image from tfrecord
            - label - coressponding label to image
            - resize_shape - list - new image shape given in [width,height] format
            Returns:
            - image - resized image
            - label - corresponding label to image
        """
        if self.resize_shape != None:
            image = tf.image.resize(image,[*self.resize_shape])
        else:
            pass 
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image,0.5)
        image = tf.image.random_contrast(image,0.2,3)
        image = tf.image.random_saturation(image,0.2,3)
        image = tfa.image.mean_filter2d(image)

        return image,label
    
    def data_augment_for_raw_jpg(self,
                                image,
                                label):
        """
            This function is responsible for augmenting training data
            obtained from raw jpg files. These files differs from images 
            in tfrecords as they were inappropriately created (blue channel is shifted with red channel)
            Args:
            - image - image from tensorflow Dataset 
            - label - corresponding label to image
            Returns:
            - image - augmented image with shifted channels (from RGB to BGR)
            - label - corresponding label to image
        """
        
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image,0.5)
        image = tf.image.random_contrast(image,0.2,3)
        image = tf.image.random_saturation(image,0.2,3)
        image = tfa.image.mean_filter2d(image)

        return image,label 

    def data_only_resize(self,
                    image,
                    label):
        """
            Perform resizing images on validation dataset (valiation dataset should not be augmented)
            Args:
            - image - image from tfrecord 
            - label - corresponding label to image
            Returns:
            - image - resized image 
            - label - corresponding label to image 
        """
        image = tf.image.resize(image[*self.resize_shape])
        return image,label

    def load_dataset(self,filename,
                          labeled=True,
                          ordered=False):
        """
            Loading dataset from tfrecord
            Args:
            - filename - filename with tfrecord extension with data to parse
            - labeled - boolean if the data is labeled or not (turn this to True when loading test files)
            - ordered - if the data is Ordered.
            Returns:
            - dataset - Loaded tensorflow dataset.
        """
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False
        
        dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(self.read_labeled_tfrecord if labeled else self.read_unlabeled_tfrecord)
        
        return dataset

    def get_training_dataset(self,
                            labeled=True,
                            ordered=False):
        """
            Read Training data as a TFDataset
        """
        dataset = self.load_dataset(self.train_files,labeled=labeled,ordered=ordered)
        dataset = dataset.map(self.data_augment_and_resize,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        dataset = dataset.repeat() 
        dataset = dataset.shuffle(self.shuffle)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE) 
        return dataset
    
    def get_validation_dataset(self,
                                labeled=True,
                                ordered=False):
        """
            Read validation data as a TFDataset
        """
        dataset = self.load_dataset(self.validation_files,labeled=labeled,ordered=ordered)
        if self.resize_shape != None:
            dataset = dataset.map(self.data_only_resize,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        else:
            pass 
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE) 
        return dataset

    
    def get_test_dataset(self,
                        labeled = False,
                        ordered = True):
        """
            Function for reading test dataset to make submission
            Args:
            - labeled - if the data is labeled. 
            - ordered - if the data is ordered.
            Returns:
            - Tensorflow dataset with test files.
        """
        dataset = self.load_dataset(self.test_files,
                               labeled=labeled,ordered=ordered)
        dataset = dataset.map(self.data_only_resize,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE)
        return dataset
    
    def fetch_train_iterator(self):
        """
            Generator that output images with dimensions [batch_size,image_width,image_height,num_channels]
            and corresponding image label. Generator provides more memory efficient iterating through batches.
            Returns:
            - inputs - images with format [batch_size,image_width,image_height,num_channels]
            - output - label with format [batch_size,num_labels]
        """
        for image,label in self.train_tfdataset:
            yield (image.numpy(),label.numpy())
        
    def fetch_valid_iterator(self):
        """
            This function is similar to fetch_train_iterator but here we generate validation samples.
            Returns: 
            - inputs - images with format [batch_size,image_width,image_height,num_channels]
            - output - label with format [batch_size,num_labels]
        """
        # Use only if you created validation dataset
        for image,label in self.validation_tfdataset:
            yield (image.numpy(),label.numpy())
    
    def get_train_steps_per_epoch(self):
        """
            Defines train step in each epoch.
        """
        return math.ceil(math.ceil(self.dataset_size*(1-self.validation_split))/self.batch_size)
    
    def get_validation_steps_per_epoch(self):
        """
            Defines validation step in each epoch
        """
        return math.ceil(math.ceil(self.dataset_size*self.validation_split)/self.batch_size)
    
    def get_train_from_tensor_slices(self):
        """
            Create TFDataset from training data (mostly convenient when making CV,
            as we can read raw data from folds and create TFDataset object).
        """
        return tf.data.Dataset.from_tensor_slices(
            (self.train_files[0],self.train_files[1])
        ).map(self.decode_image_from_raw_jpeg,num_parallel_calls=tf.compat.v2.experimental.AUTOTUNE
        ).map(self.data_augment_for_raw_jpg,num_parallel_calls=tf.compat.v2.experimental.AUTOTUNE
        ).repeat().shuffle(self.shuffle).batch(self.batch_size).prefetch(tf.compat.v2.experimental.AUTOTUNE)
    
    def get_val_from_tensor_slices(self):
        """
            Create TFDataset from validation data
        """
        return tf.data.Dataset.from_tensor_slices(
            (self.validation_files[0],self.validation_files[1])
        ).map(self.decode_image_from_raw_jpeg,num_parallel_calls=tf.compat.v2.experimental.AUTOTUNE
        ).batch(self.batch_size).cache().prefetch(tf.compat.v2.experimental.AUTOTUNE)