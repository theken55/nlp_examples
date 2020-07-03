# Document classification

This example includes scripts to train document classifier, and predict classes of documents using trained models.

Both multiclass and multi-label classifications are supported.

This example refers to both [Keras Tutorial](https://keras.io/examples/nlp/text_classification_from_scratch/) 
and [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

### Prerequisites
- Tensorflow 2.2

### Scripts
- driver.py
  - `Usage: driver.py [train|predict] [directory|file.csv]`
  - mode:
    - `train`: train model from documents
    - `predict`: predict documents by trained model
  - documents:
    - `directory`: expected folder structure is:
    ```
    train
      |-pos
         |- *.txt
      |-neg
         |- *.txt
    test
      |-pos
         |- *.txt
      |-neg
         |- *.txt
    ```
    - `file.csv`: expected format is:
    ```
    text, label
    This is a pen, label1
    I have an apple, label2
    ...
    ```
- dataio.py
  - load text into Tensorflow dataset
  - split dataset
- modelm.py
  - define two neural network model architecture
    - multiclass classification
    - multi-label classification
  - save/load SavedModel, label

### Results

#### train

```
...
782/782 [==============================] - 2s 3ms/step - loss: 0.3570 - accuracy: 0.8544
[array([b'there).', b'this).', b'thought.<br', ..., b'actors',
       b'similarity', b'"b"'], dtype=object), array([20000, 19998, 19997, ...,   184,  8740,  7005]), array([[-0.00491479, -0.04982085, -0.02650738, ..., -0.00917612,
        -0.04212022, -0.01764694],
...
```

#### predict

```
The story is a good comedy.
neg:0.340374
pos:0.659626

It is a silly story.
neg:0.610098
pos:0.389902

This isn't a very exciting film, but it's warm.
neg:0.181591
pos:0.818409
```


### Reference
- Keras: Text classification from scratch
  - https://keras.io/examples/nlp/text_classification_from_scratch/
- Large Movie Review Dataset
  - https://ai.stanford.edu/~amaas/data/sentiment/
