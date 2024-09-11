import pickle
import tensorflow as tf
import random
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer

# code below taken from @JosephElliott-kx9yj from https://www.youtube.com/watch?v=1lwddP0KUEg
# some changes were made to improve bot performance and to make it work with the website

'''creates a class so data used in multiple functions can be shared. This is not necessary, but then values
would need to be passed as parameters'''

class ChatNet:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('intents.json').read()) # loads file needed to train the model
        self.words = []
        self.classes = []
        self.documents = []

    def prepare_data(self):
        '''
        creates lists of words and 'tags' (saved as classes) that are saved to be used in the training and
        when using the model. also creates a 'documents' list. The document list has the tag for each topic as well a
        as the words for each question in its own list. I recommend printing the lists to see what they  look like.
        '''
        ignoreLetters = ['?', '!', '.', ',']
        for intent in self.intents['intents']:  # iterates through each list in the file
            for pattern in intent['patterns']:  # iterates through each question in patterns
                wordList = nltk.word_tokenize(pattern)  # tokens breaks each question into words
                self.words.extend(wordList)  # adds each word list to word file
                self.documents.append((wordList, intent['tag']))  # adds each word list and tag to documents file
                if intent['tag'] not in self.classes:  # if the tags isn't already in the classes list
                    self.classes.append(intent['tag'])

                    # a lemmatizer creates a list of words that have been broken down to their root words
        # Example: catch, catches, catching -> root word: catch
        self.words = [self.lemmatizer.lemmatize(word) for word in self.words if word not in ignoreLetters]
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        pickle.dump(self.words, open('models/words.pkl', 'wb'))  # saves the lemmatized words list to a file
        pickle.dump(self.classes, open('models/classes.pkl', 'wb'))  # saves the classes list to a file

    def bag_of_words(self):
        '''
        Creates a 'bag of words' which is just a matrix of 1s and 0s.  It compares all of the words in the entire
        document to the words from each topic. If the word is in the topic, a 1 is put into the matrix. Otherwise,
        a 0 is put into the matric.
        :return: file containing the 'bag of words' in matrix from
        '''
        training = [] # creates a list to hold the training data
        outputEmpty = [0] * len(self.classes)
        for document in self.documents:
            bag = [] #creates an empty list to hold words
            # saves the tokenized word list in the documents file to a new variable
            wordPatterns = document[0]
            # makes each word in the word list lowercase and then finds its root word, saves all words to a new list
            wordPatterns = [self.lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
            #adds words in words list to bag of words only if in wordPatterns list
            for word in self.words:
                bag.append(1) if word in wordPatterns else bag.append(0)
            outputRow = list(outputEmpty)
            outputRow[self.classes.index(document[1])] = 1
            training.append(bag + outputRow) #adds classes to training data
        return training

    def train_model(self):
        '''
        Uses the methods above to prepare the data. Then creates training lists. Then uses a neural network to
        train the chatbot model.
        :return: a model of the trained chatbot
        '''
        self.prepare_data()
        training = self.bag_of_words()
        random.shuffle(training)  # randomly change the order of the training data
        training = np.array(training)
        trainX = training[:, :len(self.words)]
        trainY = training[:, len(self.words):]

        # The lines below use a neural network to train the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))
        sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(trainX, trainY, epochs=50, batch_size=4)
        model.save('models/chatbot_model.keras')  # saves the model so it can be used in chatbot
        return model