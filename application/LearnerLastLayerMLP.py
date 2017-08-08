#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 10:41
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : LearnerLastLayerMLP
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.ModelCompletion import ModelCompletion

class LearnerLastLayerMLP(ModelCompletion):

    def __init__(self, name, dataset):
        super(LearnerLastLayerMLP, self).__init__(name, dataset)

    def learn(self):
        ground_truth = {}
        with open(GROUND_TRUTH_FILE, 'rb') as groundTruthFile:
            list = groundTruthFile.readlines()
            for row in list:
                row = re.sub(r'\n', '', row)
                buf = row.split(',')
                label_row = buf[0]
                ground_truth_row = buf[1].split('-')
                ground_truth[label_row] = ground_truth_row

        if not os.path.exists("tmp/model/model.json"):
            # create model
            model = Sequential()
            model.add(Dense(1024, input_dim=BOTTLENECK_TENSOR_SIZE, activation='relu'))
            model.add(Dense(TOTAL_NUM_CLASSES, activation='softmax'))  # softmax sigmoid

            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[
                top3_accuracy])  # accuracy 'categorical_crossentropy' 'categorical_accuracy' multiclass_loss

            # plot_model(model, filename='model.png', show_shapes=True, show_layer_names=True)

            tensorboard = keras.callbacks.TensorBoard(
                log_dir='/home/invincibleo/Downloads/tensorboardLogs',
                histogram_freq=10, write_graph=True, write_images=True)

            hist = model.fit_generator(
                generate_arrays_from_file(audio_lists, ground_truth, bottleneck_dir, 'training', batch_size),
                steps_per_epoch=1,
                epochs=100000,  # 1000000
                validation_data=generate_arrays_from_file(audio_lists, ground_truth, bottleneck_dir, 'validation',
                                                          batch_size),
                validation_steps=1,
                verbose=2,
                callbacks=[tensorboard]
            )

            if not os.path.exists("tmp/model"):
                os.makedirs("tmp/model")
            # Saving the objects:
            with open('tmp/model/objs.txt', 'wb') as histFile:  # Python 3: open(..., 'wb')
                # pickle.dump([hist, model], f)
                for key, value in hist.history.iteritems():
                    histFile.write(key + '-' + ','.join([str(x) for x in value]))
                    histFile.write('\n')
            # # Getting back the objects:
            # with open('objs.pickle') as f:  # Python 3: open(..., 'rb')
            #     hist, model = pickle.load(f)

            # scores = model.evaluate(X, Y, verbose=0)
            # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


            # serialize model to JSON
            model_json = model.to_json()
            with open("tmp/model/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("tmp/model/model.h5")
            print("Saved model to disk")
        else:
            # load json and create model
            json_file = open("tmp/model/model.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("tmp/model/model.h5")
            print("Loaded model from disk")

        return model

    def predict(self):
        pass

