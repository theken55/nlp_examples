import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def custom_standardization(input_data):
    return tf.strings.lower(input_data)

class ModelBase():
    def __init__(self):
        self.LABEL_FILE = "mylabel.txt"
        
    def save(self, model, model_path, label_map):
        model.save(model_path, save_format="tf")
        # save label
        label_path = os.path.join(model_path, self.LABEL_FILE)
        with open(label_path, "w", encoding='utf-8') as f:
            for label, _ in sorted(label_map.items(), key=lambda x: x[1]):
                f.write(label + '\n')
        
    def load(self, model_path):
        model = tf.keras.models.load_model(model_path, 
                                           custom_objects={'TextVectorization':TextVectorization, 'custom_standardization':custom_standardization})
        # load label
        label_path = os.path.join(model_path, self.LABEL_FILE)
        label_map = {}
        with open(label_path, "r", encoding='utf-8') as f:
            for _line in f.readlines():
                line = _line.rstrip()
                if len(line) > 0:
                    label_map[line] = len(label_map)
        
        return model, label_map

class MultiClassModel(ModelBase):
    def construct(self, text_ds, label_num):
        max_features = 20000
        embedding_dim = 128
        sequence_length = 200
        
        # from text to word sequence by white space at default
        # from word to word id
        vectorize_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=max_features,
            output_mode="int",
            output_sequence_length=sequence_length,
        )
        vectorize_layer.adapt(text_ds)
        inputs = tf.keras.Input(shape=(1,), dtype="string")
        indices = vectorize_layer(inputs)
        
        # from word id to word embedding. 
        # max_features + 1 for OOV
        x = layers.Embedding(max_features + 1, embedding_dim)(indices)
        x = layers.Dropout(0.5)(x)
        
        # global max pooling
        x = layers.GlobalMaxPooling1D()(x)
        predictions = layers.Dense(label_num, activation="softmax", name="predictions")(x)
        model = tf.keras.Model(inputs, predictions)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        return model
    
class MultiLabelModel(ModelBase):
    def construct(self, text_ds, label_num):
        max_features = 20000
        embedding_dim = 128
        sequence_length = 200
        
        vectorize_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=max_features,
            output_mode="int",
            output_sequence_length=sequence_length,
        )
        vectorize_layer.adapt(text_ds)
        
        inputs = tf.keras.Input(shape=(1,), dtype="string")
        indices = vectorize_layer(inputs)
        
        x = layers.Embedding(max_features + 1, embedding_dim)(indices)
        x = layers.Dropout(0.5)(x)
        
        # global max pooling
        x = layers.GlobalMaxPooling1D()(x)
        predictions = layers.Dense(label_num, activation="sigmoid", name="predictions")(x)
        model = tf.keras.Model(inputs, predictions)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        return model
