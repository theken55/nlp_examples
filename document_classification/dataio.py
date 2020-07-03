import tensorflow as tf
import csv
import glob
import os
import random
random.seed(23)

class DataIO:
    def load(self, data_path, batch_size=1, is_one_hot=False):
        label_map = {'neg' : 0, 'pos' : 1}
        LABEL_NUM = len(label_map)
        def _load(sub_data_path):
            texts = []
            labels = []
#             count = 0
            for file in glob.glob(os.path.join(sub_data_path, "neg/*.txt")):
                with open(file) as f:
                    texts.append(f.read())
                    labels.append(0)
#                     count += 1
#                     if count >= 1000:
#                         break
            for file in glob.glob(os.path.join(sub_data_path, "pos/*.txt")):
                with open(file) as f:
                    texts.append(f.read())
                    labels.append(1)
#                     count += 1
#                     if count >= 2000:
#                         break
            texts, labels = self.shuffle(texts, labels)
            text_ds = tf.data.Dataset.from_tensor_slices(texts)
            label_ds = tf.data.Dataset.from_tensor_slices(labels)
            if is_one_hot:
                label_ds = label_ds.map(lambda x: tf.one_hot(x, LABEL_NUM))
            text_ds = text_ds.batch(batch_size)
            label_ds = label_ds.batch(batch_size)
            ds = tf.data.Dataset.zip((text_ds, label_ds))
            return ds
        
        train_ds = _load(os.path.join(data_path, 'train'))
        test_ds = _load(os.path.join(data_path, 'test'))
        return train_ds, test_ds, label_map
    
    '''
    Load csv format. Expected format is:
    text, label
    This is a pen, label1
    I have an apple, label2
    ...
    '''
    def load_csv(self, file, batch_size=1, is_one_hot=False, label_map=None):
        if label_map is None:
            label_map = {}
        texts = []
        labels = []
#         label_map = {}
        with open(file) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                text = row[-2]
                label = row[-1]
                texts.append(text)
                if label not in label_map:
                    label_map[label] = len(label_map)
                labels.append(label_map[label])
        
        LABEL_NUM = len(label_map)
        texts, labels = self.shuffle(texts, labels)
        text_ds = tf.data.Dataset.from_tensor_slices(texts)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        if is_one_hot:
            label_ds = label_ds.map(lambda x: tf.one_hot(x, LABEL_NUM))
        text_ds = text_ds.batch(batch_size)
        label_ds = label_ds.batch(batch_size)
        
        ds = tf.data.Dataset.zip((text_ds, label_ds))
        train_ds, test_ds = self.split(ds)
        return train_ds, test_ds, label_map
    
    def shuffle(self, texts, labels):
        pairs = list(zip(texts, labels))
        random.shuffle(pairs)
        texts, labels = zip(*pairs)
        return list(texts), list(labels)
    
    def split(self, all_dataset, ratio=0.8):
        DATASET_SIZE = 0
        for _ in all_dataset:
            DATASET_SIZE += 1
        
        ds1_size = int(ratio * DATASET_SIZE)
        ds2_size = DATASET_SIZE - ds1_size
        print(("ds1_size:%d, ds2_size:%d") % (ds1_size, ds2_size));
        
        ds1 = all_dataset.take(ds1_size)
        ds2 = all_dataset.skip(ds1_size)
        return ds1, ds2
    
    def create_dataset(self, texts, batch_size=1):
        myds = tf.data.Dataset.from_tensor_slices(texts)
        myds = myds.batch(batch_size)
        return myds
