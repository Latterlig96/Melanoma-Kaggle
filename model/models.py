import numpy as np 
from tensorflow.keras.layers import GlobalAveragePooling2D,Multiply
from tensorflow.keras.layers import Concatenate,BatchNormalization
from tensorflow.keras.layers import Dense,Dropout,Flatten,Input,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications.nasnet import NASNetMobile,preprocess_input
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.layers import AdaptiveAveragePooling2D,AdaptiveMaxPooling2D
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
                mode = None,
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
                - mode - mode of architecture to be trained
                        f.e linear means that we use linear head,
                        non linear means that we use more dense layers with relu activation 
                        and meta means that we build model that includes both metadata and image data.
                """

                self.architecture = architecture
                self.learning_rate = learning_rate 
                self.input_shape = input_shape
                self.output_shape = output_shape
                self.optimizer = optimizer
                self.metric = metric
                self.loss = loss 
                self.model_data_dir = model_data_dir
                self.mode = mode 
                self.verbose = verbose
    
                #self.set_mixed_precision()

                if self.architecture == 'efficientnet':
                    self.model = self.create_model_efficientnet(self.mode)
                else:
                    self.model = self.create_model_nasnet(self.mode)

                if verbose:
                    print(f"Created model: {self.architecture}")
                    print(f"Input Shape: {self.input_shape}")
                    print(f"Output Shape: {self.output_shape}")
                    print("Mixed precission is set")

    def create_model_efficientnet(self,mode=None):
        """
            Create one of the efficientnet models (by now its EfficientNetB5)
            Arguments: 
            - mode - mode of architecture to be trained
                        f.e linear means that we use linear head,
                        non linear means that we use more dense layers with relu activation 
                        and meta means that we build model that includes both metadata and image data.
            Returns:
            - Model - Keras model that is ready to be fit.
        """
        if mode == 'meta':
            input_tensor = Input(shape=self.input_shape[0],name='inp1')
            input_meta = Input(shape=self.input_shape[1],name='inp2')
        else:
            input_tensor = Input(shape=self.input_shape)
        base_model = efn.EfficientNetB3(include_top=False,input_tensor=input_tensor)
        x = base_model(input_tensor)
        if mode == 'linear':
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048,activation='linear')(x)
            x = Dense(self.output_shape,activation='sigmoid')(x)
            model = Model(input_tensor,x)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=self.metric)
            model.summary()
            return model

        elif mode == 'non_linear':
            out1 = AdaptiveMaxPooling2D((2,2))(x)
            out2 = AdaptiveAveragePooling2D((2,2))(x)
            out = Concatenate(axis=-1)([out1,out2])
            out = Flatten()(out)
            out = Dense(2048,activation='relu')(out)
            out = Dropout(0.4)(out)
            out = Dense(1024,activation='relu')(out)
            out = Dropout(0.3)(out)
            out = Dense(512,activation='relu')(out)
            out = Dropout(0.2)(out)
            out = Dense(256,activation='relu')(out)
            out = Dropout(0.1)(out)
            out = Dense(self.output_shape,activation='sigmoid')(out)
            model = Model(input_tensor,out)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=self.metric)
            model.summary()
            return model
        
        elif mode == 'meta':
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048,activation='linear')(x)
            x = BatchNormalization()(x)
            x1 = Dense(128)(input_meta)
            x1 = LeakyReLU()(x1)
            x1 = BatchNormalization()(x1)
            concat = Concatenate()([x,x1])
            x = Dense(1000,activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dense(300,activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dense(80,activation='relu')(x)
            x1 = Dense(60,activation='linear')(x)
            x1 = BatchNormalization()(x1)
            x1 = Dense(20,activation='linear')(x1)
            x1 = BatchNormalization()(x1)
            concat = Dense(1300,activation='relu')(concat) 
            concat = BatchNormalization()(concat)
            concat = Dense(400,activation='relu')(concat)
            concat = BatchNormalization()(concat)
            concat = Dense(70,activation='relu')(concat)
            concat_layer = Concatenate()([x,x1,concat])
            last_head = Dense(20,activation='relu')(concat_layer)
            last_head = BatchNormalization()(last_head)
            output = Dense(1,activation='sigmoid')(last_head)
            model = Model([input_tensor,input_meta],output)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=self.metric)
            model.summary()
            return model 

    def create_model_nasnet(self,mode=None):
        """
            Creates NasNetMobile model.
            Arguments: 
            - mode - mode of architecture to be trained
                        f.e linear means that we use linear head,
                        non linear means that we use more dense layers with relu activation 
                        and meta means that we build model that includes both metadata and image data.
            Returns:
            - Model - Keras model that is ready to be fit.
        """
        if mode == 'meta':
            input_tensor = Input(shape=self.input_shape[0],name='inp1')
            input_meta = Input(shape=self.input_shape[1],name='inp2')
        else:
            input_tensor = Input(shape=self.input_shape)
        base_model = NASNetMobile(include_top=False,input_tensor=input_tensor)
        x = base_model(input_tensor)
        if mode == 'linear':
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048,activation='linear')(x)
            x = Dense(self.output_shape,activation='sigmoid')(x)
            model = Model(input_tensor,x)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=self.metric)
            model.summary()
            return model 

        elif mode == 'non_linear':
            out1 = AdaptiveMaxPooling2D((2,2))(x)
            out2 = AdaptiveAveragePooling2D((2,2))(x)
            out = Concatenate(axis=-1)([out1,out2])
            out = Flatten()(out)
            out = Dense(2048,activation='relu')(out)
            out = Dropout(0.4)(out)
            out = Dense(1024,activation='relu')(out)
            out = Dropout(0.3)(out)
            out = Dense(512,activation='relu')(out)
            out = Dropout(0.2)(out)
            out = Dense(256,activation='relu')(out)
            out = Dropout(0.1)(out)
            out = Dense(self.output_shape,activation='sigmoid')(out)
            model = Model(input_tensor,out)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=self.metric)
            model.summary()
            return model 

        elif mode == 'meta':
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048,activation='linear')(x)
            x = BatchNormalization()(x)
            x1 = Dense(128)(input_meta)
            x1 = LeakyReLU()(x1)
            x1 = BatchNormalization()(x1)
            concat = Concatenate()([x,x1])
            x = Dense(1000,activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dense(300,activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dense(80,activation='relu')(x)
            x1 = Dense(60,activation='linear')(x)
            x1 = BatchNormalization()(x1)
            x1 = Dense(20,activation='linear')(x1)
            x1 = BatchNormalization()(x1)
            concat = Dense(1300,activation='relu')(concat) 
            concat = BatchNormalization()(concat)
            concat = Dense(400,activation='relu')(concat)
            concat = BatchNormalization()(concat)
            concat = Dense(70,activation='relu')(concat)
            concat_layer = Concatenate()([x,x1,concat])
            last_head = Dense(20,activation='relu')(concat_layer)
            last_head = BatchNormalization()(last_head)
            output = Dense(1,activation='sigmoid')(last_head)
            model = Model([input_tensor,input_meta],output)
            model.compile(optimizer=self.optimizer(self.learning_rate),
                          loss=self.loss,
                          metrics=self.metric)
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