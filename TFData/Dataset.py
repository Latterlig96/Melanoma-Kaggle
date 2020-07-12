import tensorflow as tf 
import tensorflow_addons as tfa 
from tensorflow.keras.models import load_model
import math 
import numpy as np
import pandas as pd  
import random
import os 
class Dataset:

    def __init__(self,
                train_files = None,
                test_files = None,
                validation_files = None,
                validation_split = None,
                image_size = None,
                dataset_size = None,
                batch_size = None,
                shuffle = None,
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
        self.shuffle = shuffle
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
                                example,
                                with_meta = True,
                                ):
        """
           In this function you read images nad labels from tfrecord in their
           default data type and parse them to a format where they can be 
           fed into model.
           Args:
           - example - tfrecord label contains image and corresponding label
           - with_meta - bool - whether to read the metadata features
           Returns:
           - Image - image casted as a tf.float32 tensor object with size [1024,1024,3]
           - label - image corresponding label in tf.int32 format 
        """
        if with_meta:
            feature = {
                "image": tf.io.FixedLenFeature([],tf.string),
                "sex": tf.io.FixedLenFeature([], tf.int64),
                "age_approx": tf.io.FixedLenFeature([],tf.int64),
                "anatom_site_general_challenge": tf.io.FixedLenFeature([],tf.int64),
                "diagnosis": tf.io.FixedLenFeature([],tf.int64),
                "target": tf.io.FixedLenFeature([],tf.int64)
                }
        else: 
            feature = {
            "image": tf.io.FixedLenFeature([],tf.string),
            "target": tf.io.FixedLenFeature([],tf.int64)
                    }
        example = tf.io.parse_single_example(example,feature)

        if with_meta:
            data = {} 
            data['sex'] = tf.cast(example['sex'],tf.int32)
            data['age_approx'] = tf.cast(example['age_approx'],tf.int32)
            data['anatom_site_general_challenge'] = tf.cast(example['anatom_site_general_challenge'],tf.int32)
            data['diagnosis'] = tf.cast(example['diagnosis'],tf.int32)
        image = self.decode_image(example['image'])
        label = tf.cast(example['target'],tf.int32)

        if with_meta:
            return image,label,data
        else:
            return image,label

    
    def read_unlabeled_tfrecord(self,
                                example,
                                with_meta = True):
        """
            This function is similiar to a read_labeled_tfrecord function 
            but here instead of having label we have image_name.
            Args:
            - example - tfrecord label contains image and coressponding label
            - with_meta - whether to read metadata features
            Returns:
            - Image - image casted as a tf.float32 tensor object with size [1024,1024,3]
            - idnum - image name corresponding to parsed image
        """
        if with_meta:
            feature = {
                "image": tf.io.FixedLenFeature([],tf.string),
                "image_name": tf.io.FixedLenFeature([],tf.string),
                "sex": tf.io.FixedLenFeature([],tf.int64),
                "age_approx": tf.io.FixedLenFeature([],tf.int64),
                "anatom_site_general_challenge": tf.io.FixedLenFeature([],tf.int64),    
                    }
        else:
            feature = {
            "image": tf.io.FixedLenFeature([],tf.string),
            "image_name": tf.io.FixedLenFeature([],tf.string),    
                }
        example = tf.io.parse_single_example(example,feature)

        if with_meta:
            data = {} 
            data['sex'] = tf.cast(example['sex'],tf.int32)
            data['age_approx'] = tf.cast(example['age_approx'],tf.int32)
            data['anatom_site_general_challenge'] = tf.cast(example['anatom_site_general_challenge'],tf.int32)
        image = self.decode_image(example['image'])
        idnum = example['image_name']

        if with_meta: 
            return image,idnum,data
        else: 
            return image,idnum
    
    def read_full_tfrecord(self,
                            example):
        """
            Read full tfcored from given examples, this function is similar to 
            read_labeled_tfcored but given that we must provide object when reading 
            mapping functions it is not possible to sneak around and try to load 
            f.e image_name because it will raise an error, so we must provide different function.
        """

        feature = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "image_name": tf.io.FixedLenFeature([], tf.string), 
        "target": tf.io.FixedLenFeature([], tf.int64), 
        "diagnosis": tf.io.FixedLenFeature([],tf.int64),
        "age_approx": tf.io.FixedLenFeature([], tf.int64),
        "sex": tf.io.FixedLenFeature([], tf.int64),
        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64)
                }

        example = tf.io.parse_single_example(example, feature)
        image = decode_image(example['image'])
        image_name = example['image_name']
        label = tf.cast(example['target'], tf.float32)
        data = {}
        data['age_approx'] = tf.cast(example['age_approx'], tf.int32)
        data['diagnosis'] = tf.cast(example['diagnosis'], tf.int32)
        data['sex'] = tf.cast(example['sex'], tf.int32)
        data['anatom_site_general_challenge'] = tf.cast(tf.one_hot(example['anatom_site_general_challenge'], 7), tf.int32)

        return image, image_name, label, data
    
    def train_data_setup(self,
                        image,
                        label,
                        data):
        """
            Setup training data to float32 (which is deafult datatype when training)
            Args: 
            - image - image to train 
            - label - coressponding label to image 
            - data - metadata features
            Returns: 
            Dict - dictionary of different inputs which can be fed into model 
            label - corresponding label to Dict data
        """
        anatom = [tf.cast(data['anatom_site_general_challenge'],tf.float32)]
        diagnosis = [tf.cast(data['diagnosis'],tf.float32)]

        tab_data = [tf.cast(data[feat],tf.float32) for feat in ['age_approx','sex']]

        tabular = tf.stack(tab_data + anatom + diagnosis)

        return {'inp1': image, 'inp2': tabular}, label
    
    def test_data_setup(self,
                        image,
                        idnum,
                        data):
        """
            Setup test data to float32 
            Args: 
            - image - image to train 
            - idnum - corresponding image name 
            - data - metadata features
            Returns: 
            Dict - dictionary of different inputs which can be fed into model 
            idnum - coressponding image name to data 
        """
        anatom = [tf.cast(data['anatom_site_general_challenge'],tf.float32)]

        tab_data = [tf.cast(data[feat],tf.float32) for feat in ['age_approx','sex']]

        tabular = tf.stack(tab_data + anatom)

        return {'inp1': image,'inp2': tabular}, idnum
    
    def full_data_setup(self,
                        image,
                        image_name,
                        label,
                        data):
        """
            Setup full data to float32 
            Args: 
            - image - image to train 
            - image_name - image name of given example 
            - label - corresponding label to image
            - data - metadata features
        """
        anatom = [tf.cast(data['anatom_site_general_challenge'],tf.float32)]
        diagnosis = [tf.cast(data['diagnosis'],tf.float32)]

        tab_data = [tf.cast(data[feat],tf.float32) for feat in ['age_approx','sex']]

        tabular = tf.stack(tab_data + anatom + diagnosis)

        return {'inp1': image, 'inp2': tabular}, image_name, label


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
        image = tf.image.random_brightness(image,0.1)
        image = tf.image.random_contrast(image,0.2,3)
        image = tf.image.random_saturation(image,0.2,3)

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
        image = tf.image.random_brightness(image,0.1)
        image = tf.image.random_contrast(image,0.2,3)
        image = tf.image.random_saturation(image,0.2,3)
        
        return image,label 

    def data_augment_and_resize_with_meta(self,
                                          data,
                                          label):
        """
            Similiar function to data_augment_and_resize
            but here we operate mostly on dict that consist of 
            image and metadata features.
            Args: 
            - data - dict consist of images given with key inp1 
            and metadata features with key inp2
            - label - label corresponding to given example
            Returns: 
            - data - dict with augmented images (no hierarchy is interrupted)
            - label - corresponding label to given example
        """
        if self.resize_shape != None:
            data['inp1'] = tf.image.resize(data['inp1'],[*self.resize_shape])
        else:
            pass 
        data['inp1'] = tf.image.random_flip_left_right(data['inp1'])
        data['inp1'] = tf.image.random_flip_up_down(data['inp1'])
        data['inp1'] = tf.image.random_brightness(data['inp1'],0.1)
        data['inp1'] = tf.image.random_contrast(data['inp1'],0.2,3)
        data['inp1'] = tf.image.random_saturation(data['inp1'],0.2,3)

        return data,label

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
        image = tf.image.resize(image,[*self.resize_shape])
        return image,label
    
    def data_only_resize_with_meta(self,
                                   data,
                                   label):
        """
            Similar function to data_only_resize but works on dict with images and metadata
        """
        data['inp1'] = tf.image.resize(data['inp1'],[*self.resize_shape])
        return data,label 

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
    
    def load_full_dataset(self,filenames):
        """
            Read full dataset, especially necessary when dealing with Folds stored 
            as tfrecords, because the data will be shuffled so image names will be 
            randomly selected and we have to find them by their image names 
            (not just index as it can be done with normal f.e. KFold)
        """

        dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.read_full_tfrecord,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)

        return dataset

    def get_training_dataset(self,
                            labeled=True,
                            ordered=False,
                            with_meta=True):
        """
            Read Training data as a TFDataset
        """
        dataset = self.load_dataset(self.train_files,labeled=labeled,ordered=ordered)
        if with_meta: 
            dataset = dataset.map(self.train_data_setup,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
            dataset = dataset.map(self.data_augment_and_resize_with_meta,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(self.data_augment_and_resize,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        dataset = dataset.repeat() 
        dataset = dataset.shuffle(self.shuffle)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE) 

        return dataset
    
    def get_validation_dataset(self,
                                labeled=True,
                                ordered=False,
                                with_meta = True):
        """
            Read validation data as a TFDataset
        """
        dataset = self.load_dataset(self.validation_files,labeled=labeled,ordered=ordered)
        if self.resize_shape != None:
            if with_meta:
                dataset = dataset.map(self.train_data_setup,num_parallel_calls = tf.compat.v2.data.experimental.AUTOTUNE)
                dataset = dataset.map(self.data_only_resize_with_meta,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
            else:
                dataset = dataset.map(self.data_only_resize,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        else:
            pass 
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE) 

        return dataset

    
    def get_test_dataset(self,
                        labeled = False,
                        ordered = True,
                        with_meta = True):
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
        if with_meta: 
            dataset = dataset.map(self.test_data_setup,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
            dataset = dataset.map(self.data_only_resize_with_meta,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(self.data_only_resize,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE)

        return dataset
    
    def get_full_dataset(self,filenames):
        """
            Function compatible with load_full_dataset (but not with other functions)
        """
        
        dataset = self.load_full_dataset(filenames)
        dataset = dataset.map(self.full_data_setup,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE)

        return dataset
    

    
    def fetch_train_iterator(self,data):
        """
            Generator that output images with dimensions [batch_size,image_width,image_height,num_channels]
            and corresponding image label. Generator provides more memory efficient iterating through batches.
            Returns:
            - inputs - images with format [batch_size,image_width,image_height,num_channels]
            - output - label with format [batch_size,num_labels]
        """
        while True:
            yield from data
        
    def fetch_valid_iterator(self,data):
        """
            This function is similar to fetch_train_iterator but here we generate validation samples.
            Returns: 
            - inputs - images with format [batch_size,image_width,image_height,num_channels]
            - output - label with format [batch_size,num_labels]
        """
        # Use only if you created validation dataset
        while True:
            yield from data
    
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
        dataset = (tf.data.Dataset.from_tensor_slices((self.train_files[0],self.train_files[1])))
        dataset = dataset.map(self.decode_image_from_raw_jpeg,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.data_augment_for_raw_jpg,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        dataset = dataset.repeat().shuffle(self.shuffle).batch(self.batch_size)
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE)

        return dataset
    
    def get_val_from_tensor_slices(self):
        """
            Create TFDataset from validation data
        """
        dataset = (tf.data.Dataset.from_tensor_slices((self.validation_files[0],self.validation_files[1])))
        dataset = dataset.map(self.decode_image_from_raw_jpeg,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).cache()
        dataset = dataset.prefetch(tf.compat.v2.data.experimental.AUTOTUNE)

        return dataset

class Dataset_TTA(Dataset):

    def __init__(self,
                test_files,
                dataset_size = 10982,
                image_size = [1024,1024],
                resize_shape = [256,256],
                batch_size = 8):
        super(Dataset_TTA,self).__init__(test_files=test_files,
                                        dataset_size = dataset_size,
                                        image_size=image_size,
                                        resize_shape=resize_shape,
                                        batch_size=batch_size)
        
        self.dataset_size = dataset_size

    def tta_augmentation(self,image,idnum):
        """
            Function reponsible for set of operations that are randomly done during TTA process
        """
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_contrast(image,0.1,0.3)

        return image,idnum

    def get_test_dataset(self,tta_augmentation = True):
        """
            Function calls a function from upper class to load test dataset
            with additional TTA augmentation 
            Args:
            tta_augmentation - whether to apply tta augmentation
            Returns: 
            dataset - TFDataset
        """
        if tta_augmentation:
            dataset = super().get_test_dataset().map(self.tta_augmentation,num_parallel_calls=tf.compat.v2.data.experimental.AUTOTUNE)
            return dataset
        else: 
            return super().get_test_dataset()
    
    def apply_tta(self,
                 models,
                 write_to_submission,
                 tta_steps=3):
        """
            Performing TTA (Test Time Augmentation) 
            Args: 
            models - list - list containing path to models (preferably given in a glob)
            tta_steps - int - number of steps in which tta will be performed (at the end average of
            predictions is computed)
            Returns: 
            final_predictions - averaged predictions made by each model given in models argument
            and their predictions in tta steps.
        """
        predictions = [] 
        for model in models:
            print(f"Loading model: {model}")
            loaded_model = load_model(model)
            stacked_predictions = []
            for i in range(tta_steps):
                dataset = self.get_test_dataset()
                images = dataset.map(lambda image,idnum: image)
                probabilities = np.concatenate(loaded_model.predict(images))
                stacked_predictions.append(probabilities)
            preds = np.stack(stacked_predictions,0).mean(0)
            predictions.append(preds)
        final_predictions = np.stack(predictions,0).mean(0)

        if write_to_submission:
            test_ids_ds = self.get_test_dataset().map(lambda image,idnum:idnum).unbatch()
            test_ids = next(iter(test_ids_ds.batch(self.dataset_size))).numpy().astype('U')
            self.output_submission(submission_path='./Dataset/sample_submission.csv',
                                  filenames = test_ids,
                                  target = final_predictions,
                                  filename='submission.csv')
                                  
        return final_predictions
    

    def submit_score_without_tta(self,
                                models,
                                write_to_submission):
        """
            Submit model predictions without TTA
            Args: 
            models - models path (given as a regex pattern)
            write_to_submission - boolean - whether to write to submission
            Returns: 
        """
        predictions = [] 
        for model in models:
            loaded_model = load_model(model)
            dataset = self.get_test_dataset(tta_augmentation=False)
            images = dataset.map(lambda image,idnum: image)
            probabilities = np.concatenate(loaded_model.predict(images))
            predictions.append(probabilities)
        
        final_predictions = np.stack(predictions,0).mean(0)

        if write_to_submission:
            test_ids_ds = self.get_test_dataset(tta_augmentation=False).map(lambda image,idnum: idnum).unbatch()
            test_ids = next(iter(test_ids_ds.batch(self.dataset_size))).numpy().astype('U')
            self.output_submission(submission_path='./Dataset/sample_submission.csv',
                                   filenames = test_ids,
                                   target = final_predictions,
                                   filename = 'submission.csv')
        
        return final_predictions

    def output_submission(self,
                        submission_path,
                        filenames,
                        target,
                        filename):
        """
            Function responsible for writing output submission ready to send.
            Args: 
            submission_path - path where the submission file lays (with csv extension)
            filenames - name of test filenames (just to reject any unwanted random occurences)
            target - target values predicted by models 
            filename - name of the final filename (with csv extension)
            Returns:
            None
        """
        submission = pd.read_csv(submission_path)
        pred_df = pd.DataFrame({'image_name':filenames,'target':target})
        submission.drop('target',inplace=True,axis=1)
        submission = submission.merge(pred_df,on='image_name')
        submission.to_csv(filename,index=False)
        return