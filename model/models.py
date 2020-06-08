import numpy as np 
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate,GlobalMaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Flatten,Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications.nasnet import NASNetMobile,preprocess_input
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import os 
import efficientnet.tfkeras as efn


class ModelCreation:
    
    def __init__(self,
                architecture,
                learning_rate,
                input_shape,
                output_shape,
                optimizer,
                metric,
                loss,
                model_data_dir='./saved_models/',
                linear = None,
                verbose = False):

                """
                Arguments: 
                - architecture - name of model architecture to run 
                - learning_rate - learning_rate to initialize training
                - input_shape - image input shape. 
                - output_shape - shape of the output
                - optimizer - tensorflow compatibile optimizer
                - metric - tensorflow compatibile metric
                - loss - tensorflow loss compatible loss 
                - model_data_dir - data dir to save model/weights
                - linear - whether to use linear activation function 
                            and try another architecture
                """

                self.architecture = architecture
                self.learning_rate = learning_rate 
                self.input_shape = input_shape
                self.output_shape = output_shape
                self.optimizer = optimizer
                self.metric = metric
                self.loss = loss 
                self.model_data_dir = model_data_dir
                self.linear = linear 
                self.verbose = verbose
    
                self.set_mixed_precision()

                if self.architecture == 'efficientnet':
                    self.model = self.create_model_efficientnet(self.linear)
                else:
                    self.model = self.create_model_nasnet(self.linear)

                if verbose:
                    print(f"Created model: {self.architecture}")
                    print(f"Input Shape: {self.input_shape}")
                    print(f"Output Shape: {self.output_shape}")
                    print("Mixed precission is set")

    def create_model_efficientnet(self,linear=None):
        """
            Create one of the efficientnet models (by now its EfficientNetB5)
            Arguments: 
            - linear - whether to use linear activation and change the architecture
            a little bit (just to check other solution in architecture building).
            Returns:
            - Model - Keras model that is ready to be fit.
        """
        input_tensor = Input(shape=self.input_shape)
        base_model = efn.EfficientNetB5(include_top=False,input_tensor=input_tensor)
        x = base_model(input_tensor)
        if linear:
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048,activation='linear')(x)
            x = Dense(self.output_shape,activation='sigmoid')(x)
            model = Model(input_tensor,x)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=[self.metric])
            model.summary()
            return model
        else:
            out1 = GlobalAveragePooling2D()(x)
            out2 = GlobalMaxPooling2D()(x)
            out3 = Flatten()(x)
            out = Concatenate(axis=-1)([out1,out2,out3])
            out = Dropout(0.5)(out)
            out = Dense(self.output_shape,activation='sigmoid')(out)
            model = Model(input_tensor,out)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=[self.metric])
            model.summary()
            return model

    def create_model_nasnet(self,linear=None):
        """
            Creates NasNetMobile model.
            Arguments: 
            - linear - whether to use linear activation and change the architecture
            a little bit (just to check other solution in architecture building).
            Returns:
            - Model - Keras model that is ready to be fit.
        """
        input_tensor = Input(shape=self.input_shape)
        base_model = NASNetMobile(include_top=False,input_tensor=input_tensor)
        x = base_model(input_tensor)
        if linear:
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048,activation='linear')(x)
            x = Dense(self.output_shape,activation='sigmoid')(x)
            model = Model(input_tensor,x)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=[self.metric])
            model.summary()
            return model 
        else:
            out1 = GlobalAveragePooling2D()(x)
            out2 = GlobalMaxPooling2D()(x)
            out3 = Flatten()(x)
            out = Concatenate(axis=-1)([out1,out2,out3])
            out = Dropout(0.5)(out)
            out = Dense(self.output_shape,activation='sigmoid')(out)
            model = Model(input_tensor,out)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=[self.metric])
            model.summary()
            return model 
    
    def save_model(self,filename=""):
        """
            Save trained model and his weights
            Arguments: 
            - filename - name of the file where model and weights will be saved
            Returns:
            None
        """
        model_file_path = self.model_data_dir + filename + self.architecture
        i = 1
        while os.path.isfile(model_file_path):
            model_file_path = self.model_data_dir + filename + self.architecture + '-' + str(i)
            i += 1
        weights_path = model_file_path + "-weights"
        model_file_path = model_file_path + '-model'
        self.model.save(model_file_path)
        self.model.save_weights(weights_path, save_format='tf')

    def set_mixed_precision(self):
        """
            This function is responsible for creating mixed precision policy,
            which allows us to train model on float16 data and predict on float32,
            so that in training model will consume less memory.
        """
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    
    def inject_callbacks(self,callbacks):
        """
            Injecting callbacks to model.
            Args:
            - callbacks - callbacks to be injected to model 
            Returns:
            - list of callbacks.
        """
        self.callbacks = [] 
        for callback in callbacks:
            self.callbacks.append(callbacks)
        return self.callbacks