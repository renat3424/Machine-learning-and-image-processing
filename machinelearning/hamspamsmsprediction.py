import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
import numpy as np
from collections import Counter
from tensorflow.keras import layers
import tensorflow as tf
# open the f i l e in the directory
file = open("E:\\второй компьютер\\Диск E\\sms.txt", "r", encoding="utf-8")
line = file.readline()
target = []
data = []
while 1:
    # split each line in a list , the delimiters are removed
    line = file.readline().split()
    # stop reading when the len of the line is equal to zero
    if len(line) == 0:
        break
    target.append(line[0])
    del (line[0])
    data.append(line)
file.close()


def create_dictionary(traindir):
    allwords = []
    # create a list of all words
    for i in range(len(data)):
        for j in range(len(data[i])):
            allwords += data[j]
    print(len(allwords))
    dictionary = Counter(allwords)

    # Code for non-word removal here0
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    # Sorting most commun words
    dictionary = dictionary.most_common(3000)
    return dictionary


dictionary = create_dictionary(data)

list_data = []
vector_data = []
for i in range(len(dictionary)):
    list_data.append(dictionary[i][0])

### Data vector's creation
for k in range(len(data)):
    vector = np.zeros(891)  # 891 words in dictionary after sorting it
    for j in range(len(data[k])):
        for i in range(890):
            if data[k][j] == list_data[i]:
                vector[i] = vector[i] + 1
                break
    vector_data.append(vector)

### Label vector's creation  ham=1 , spam=0
target_vector = np.zeros(5573)
for i in range(len(target)):
    if target[i] == 'ham':
        target_vector[i] = 1
    elif target[i] == 'spam':
        target_vector[i] = 0

    ### Prediction vector's creation

target_pred = np.zeros(4000)
target_test = np.zeros(1573)
target_pred[0:4000] = target_vector[0:4000]
target_test[0:1573] = target_vector[4000:5573]

data_pred = []
data_test = []
data_pred[0:4000] = vector_data[0:4000]
data_test[0:1573] = vector_data[4000:5573]

target_pred=np.reshape(target_pred, (len(target_pred), 1))
target_test=np.reshape(target_test, (len(target_test), 1))
model = Sequential()
model.add(Dense(512, input_shape=(891,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.array(data_pred), target_pred, batch_size=32, epochs=2, verbose=1, validation_split=0.1)
score=model.evaluate(np.array(data_test), target_test, batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])