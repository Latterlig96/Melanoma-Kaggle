import tensorflow as tf 
import math 

class Dataset:

    def __init__(self,
                train_files,
                test_files,
                validation_split,
                image_size,
                dataset_size,
                batch_size):
        
        """
           Args:
           - train_files - tfrecord file that contains training data 
           NOTE: remember to use glob to get all the files from the train directory.
           - test_files - tfrecord file that contains test data 
           - validation_split - [0,1] - parameters that tells how much data you want have in validation dataset
           - image_size - image_size in tfrecord
           - dataset_size - number of examples in trainingset
           - batch_size - height of batch_size
        """
        
        self.train_files = train_files
        self.test_files = test_files
        self.validation_split = validation_split 
        self.image_size = image_size
        self.dataset_size = dataset_size
        self.batch_size = batch_size 

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
                    label,
                    resize_shape = [512,512]):
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
        image = tf.image.random_flip_left_right(image)
        image = tf.image.adjust_brightness(image,0.3)
        if resize_shape != None:
            image = tf.image.resize(image,[*resize_shape])
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
        dataset = dataset.shuffle(2048)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(self.read_labeled_tfrecord if labeled else self.read_unlabeled_tfrecord)
        
        return dataset
    
    def get_train_and_validation_dataset(self):
        """
            This function is reponsible for splitting data into train and validation sets
            from the tfrecord.
            Returns:
            This function return nothing but the validation_tfdataset and train_tfdataset
            will be initialized by this function so can be further used.
        """
        # Load dataset 
        dataset = self.load_dataset(self.train_files)
        # Creates Train and validation dataset 
        self.validation_tfdataset = dataset.take(math.ceil(self.validation_split*self.dataset_size))
        self.validation_tfdataset = self.validation_tfdataset.map(self.data_augment_and_resize)
        self.validation_tfdataset = self.validation_tfdataset.repeat() 
        self.validation_tfdataset = self.validation_tfdataset.batch(self.batch_size)
        self.validation_tfdataset = self.validation_tfdataset.prefetch(self.batch_size)
        # Create train dataset 
        self.train_tfdataset = dataset.skip(math.ceil(self.validation_split*self.dataset_size))
        self.train_tfdataset = self.train_tfdataset.map(self.data_augment_and_resize)
        self.train_tfdataset = self.train_tfdataset.repeat()
        self.train_tfdataset = self.train_tfdataset.batch(self.batch_size)
        self.train_tfdataset = self.train_tfdataset.prefetch(self.batch_size)
    
    def get_training_dataset(self):
        """
            Get training set from whole training dataset
            Returns:
            - dataset - tensorflow dataset with images and labels
        """
        # Get whole training dataset
        dataset = self.load_dataset(self.train_files)
        dataset = dataset.map(self.data_augment_and_resize)
        dataset = dataset.repeat() 
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size)

        return dataset
    
    def fetch_train_iterator(self):
        """
            Generator that output images with dimensions [batch_size,image_width,image_height,num_channels]
            and corresponding image label. Generator provides more memory efficient iterating through batches.
            Tensorflow session must be provided to run the generator
            Returns:
            - inputs - images with format [batch_size,image_width,image_height,num_channels]
            - output - label with format [batch_size,num_labels]
        """
        train_iterator = self.train_tfdataset.make_one_shot_iterator()
        fetch_values = train_iterator.get_next()

        with tf.Session().as_default() as sees: 
            while True: 
                *inputs, output = sees.run(fetch_values)
                yield inputs,output
        
    def fetch_valid_iterator(self):
        """
            This function is similar to fetch_train_iterator but here we generate validation samples.
            Returns: 
            - inputs - images with format [batch_size,image_width,image_height,num_channels]
            - output - label with format [batch_size,num_labels]
        """
        # Use only if you created validation dataset
        valid_iterator = self.validation.tfdataset.make_one_shot_iterator()
        fetch_values = valid_iterator.get_next()

        with tf.Session().as_default() as sees: 
            while True: 
                *inputs, output = sees.run(fetch_values)
                yield inputs,output
    
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