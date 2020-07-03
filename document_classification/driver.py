import sys
import os
import dataio
import modelm

NETWORK_MULTICLASS = 0
NETWORK_MULTILABEL = 1
NETWORK=NETWORK_MULTICLASS
#NETWORK=NETWORK_MULTILABEL
batch_size = 32
epochs = 3

if len(sys.argv) < 3:
    print ('Usage: driver.py [train|predict] [directory|file.csv]')
    sys.exit(-1)

mode = sys.argv[1]
data_path = sys.argv[2]

if NETWORK == NETWORK_MULTICLASS:
    model_path = "_multiclass_model_" + os.path.basename(data_path)
else:
    model_path = "_multilabel_model_" + os.path.basename(data_path)

def print_ds(ds):
    for elem in ds.take(3):
        print(elem)

if mode == 'train':
    dio = dataio.DataIO()
    if os.path.isdir(data_path):
        train_ds, test_ds, label_map = dio.load(data_path, batch_size, is_one_hot=True)
    elif data_path.endswith(".csv"):
        train_ds, test_ds, label_map = dio.load_csv(data_path, batch_size, is_one_hot=True)
    else:
        print("unsupported data_path:%s" % data_path)
        sys.exit(1)
    
    text_ds = train_ds.map(lambda x, y: x)
    train_ds, val_ds = dio.split(train_ds, ratio=0.9)
    LABEL_NUM = len(label_map)
    
    if NETWORK == NETWORK_MULTICLASS:
        mm = modelm.MultiClassModel()
    elif NETWORK == NETWORK_MULTILABEL:
        mm = modelm.MultiLabelModel()
    
    model = mm.construct(text_ds, LABEL_NUM)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.evaluate(test_ds)
    print(model.get_weights())
#    model.summary()
    mm.save(model, model_path, label_map)
    
#     print("loading")
#     model2 = mm.load(model_path)
#     print(model2.get_weights())
#     model2.evaluate(test_ds)
elif mode == 'predict':
    if NETWORK == NETWORK_MULTICLASS:
        mm = modelm.MultiClassModel()
        model, label_map = mm.load(model_path)
    elif NETWORK == NETWORK_MULTILABEL:
        mm = modelm.MultiLabelModel()
        model, label_map = mm.load(model_path)
    else:
        print("unsupported network:%d" % NETWORK)
        sys.exit(1)
    
    # predict raw documents
    dio = dataio.DataIO()
    if os.path.isdir(data_path):
        train_ds, test_ds, _ = dio.load(data_path)
    elif data_path.endswith(".csv"):
        train_ds, test_ds, _ = dio.load_csv(data_path, label_map=label_map)
    else:
        print("unsupported data_path:%s" % data_path)
        sys.exit(1)
    
    label_map_inv = {v: k for k, v in label_map.items()}
    print_ds(test_ds)
    test_ds_wo_label_ds = test_ds.map(lambda x, y: x)
    
    if NETWORK == NETWORK_MULTICLASS:
        # check accuracy from prediction manually
        result = model.predict(test_ds_wo_label_ds)
        result = result.argmax(axis=1)
        
        ret = [label_map_inv[elem] for elem in result]
        
        exp_ds = test_ds.map(lambda x, y: y)
#        print_ds(exp_ds)
        exp_np = exp_ds.as_numpy_iterator()
        corr = 0
        for act, exp in zip(ret, exp_np):
            if act == label_map_inv[exp[0]]:
                corr += 1
        print("accuracy:%f" % (corr * 100 / len(result)))
        
        mytexts = ["The story is a good comedy.", "It is a silly story.", "This isn't a very exciting film, but it's warm."]
        myds = dio.create_dataset(mytexts)
        myresult = model.predict(myds)
        for idx, mytext in enumerate(mytexts):
            print(mytext)
            for lidx, prob in enumerate(myresult[idx]):
                print("%s:%f" % (label_map_inv[lidx], prob))
            print("\n")
        
    elif NETWORK == NETWORK_MULTILABEL:
        # report results
        result = model.predict(test_ds_wo_label_ds)
        print(result[:10])
        exp_ds = test_ds.map(lambda x, y: y)
        exp_np = exp_ds.as_numpy_iterator()
        for act, exp in zip(result, exp_np):
            print(exp)
            print(act)
else:
    print("unsupported mode:%s" % mode)

