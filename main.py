import os
import sys
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

##################################################################################################################
############################## > GLOBAL VARIABLES < ##############################################################
##################################################################################################################

# ACTIONS PARAMS
TODO = [ 
    "clear", 
    "load", 
    "train", 
    "predict" 
]
WITHOUT_PRINTS = False

# PREDICTIONS PARAMS
PREDICTION_SENTANCE = "SMS SERVICES. for your inclusive text credits, pls goto www.comuk.net login= 3qxj9 unsubscribe with STOP, no extra charge. help 08702840625.COMUK. 220-CM2 9AE"

# LOADER PARAMS
MIN_AUTHORIZED_FREQUENCY=0.01
MAX_AUTHORIZED_FREQUENCY=1.0

# TRAIN PARAMS
TRAIN_RATIO = 0.7
TOTAL_EPOCHS = 50
BATCH_SIZE = 256
STEP = 0.0005

# MODEL PARAMS
LAYERS = [ 100, 50, 1 ]
DROPOUT = 0.1

# PATHS PARAMS
DATA_FOLDER_NAME = "./"
DATA_FILE_NAME = "data.csv"
VECTORIZER_FOLDER_NAME = "./vectorizer/"
VECTORIZER_FILE_NAME = "vocab.st"
MODEL_FOLDER_NAME = "./models/"
MODEL_FILE_NAME = "model.pt"
PLOTS_FOLDER_NAME = "./plots/"

# CUDA PARAMS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################################################################
############################## > LOAD DATA < #####################################################################
##################################################################################################################

""" Data to study """
class DataLoader():

    def __init__(self):
        """ Instantiate the data """

        print("\n > Loading csv data from {}...".format(DATA_FOLDER_NAME + DATA_FILE_NAME)) 
        
        self._df = pd.read_csv(DATA_FOLDER_NAME + DATA_FILE_NAME, delimiter=";", quotechar="`", converters={'Message': str, 'Category': str})
        self.inputs = self._df['Message']
        self.targets = self._df['Category'].apply(lambda x: 1 if x == "spam" else 0)

""" Traduce the string data to a vector """
ENGLISH_STEMMER=nltk.stem.SnowballStemmer('english')
class Vectorizer(CountVectorizer):

    def build_analyzer(self):
        """ Build the vectorizer wrapper """

        analyzer = super(Vectorizer, self).build_analyzer()
        return lambda doc: (ENGLISH_STEMMER.stem(w) for w in analyzer(doc))

    @staticmethod
    def Load(dataLoader : DataLoader):
        """ Load and save the vectorizer wrapper """
    
        print("\n > Loading vectorizer...")

        vectorizer = Vectorizer(stop_words='english', min_df=MIN_AUTHORIZED_FREQUENCY, max_df=MAX_AUTHORIZED_FREQUENCY)
        vectorizer.fit(dataLoader.inputs)
                
        print("Saving vectorizer in {}...".format(VECTORIZER_FOLDER_NAME + VECTORIZER_FILE_NAME))
        if not os.path.exists(VECTORIZER_FOLDER_NAME):
            print("Creating vectorizer folder {}...".format(VECTORIZER_FOLDER_NAME))
            os.makedirs(VECTORIZER_FOLDER_NAME)
        
        with open(VECTORIZER_FOLDER_NAME + VECTORIZER_FILE_NAME, 'wb') as f:
            pickle.dump(vectorizer, f)

        return vectorizer

    @staticmethod
    def Get():
        """ Get the vectorizer wrapper """
    
        print("\n > Getting vectorizer from '{}'...".format(VECTORIZER_FOLDER_NAME + VECTORIZER_FILE_NAME))
        with open(VECTORIZER_FOLDER_NAME + VECTORIZER_FILE_NAME, 'rb') as f:
            vectorizer = pickle.load(f)
            return vectorizer, len(vectorizer.vocabulary_)

""" Split the data into training and testing """
class Datasets():

    def __init__(self, dataLoader : DataLoader, vocab : Vectorizer):

        print("\n > Loading datasets...")
        
        X = torch.IntTensor(vocab.transform(dataLoader.inputs).toarray()).to(DEVICE)
        Y = torch.IntTensor(dataLoader.targets).to(DEVICE)

        # Make a couple of x, y
        size = len(X)
        allData = [ (X[i], Y[i]) for i in range(size) ]

        # Split the data
        limit = int(size * TRAIN_RATIO)
        learningDataset, testingDataset = allData[:limit], allData[limit:]

        # Use data loader to make batches
        self.learningLoader = torch.utils.data.DataLoader(learningDataset, BATCH_SIZE)
        self.testingLoader = torch.utils.data.DataLoader(testingDataset, BATCH_SIZE)

##################################################################################################################
############################## > PLOT < ##########################################################################
##################################################################################################################

class SmartPlot:
    
    def __init__(self, title="smart_plot", x_label="x_label", y_label="y_label", output_path="./output"):
        """ Creates the plot """
        
        self._data = {}
        self._title = title
        self._x_label = x_label
        self._y_label = y_label
        self._output_path = output_path

    def addPoint(self, label : str, color : str, value : float):
        """ Add a point to the plot """

        if label not in self._data:
            self._data[label] = { "color": color, "data": [] }

        self._data[label]["data"].append(value)

    def build(self):
        """ Add the title, legend, labels and store the plot in a file """
    
        print("\n > Build the plot in {}...".format(self._output_path))

        self._fig, self._ax = plt.subplots()

        for label, k in self._data.items():
            self._ax.plot(range(1, 1+len(k["data"])), k["data"], label=label, color=k["color"])

        self._ax.set_xlabel(self._x_label)
        self._ax.set_ylabel(self._y_label)
        self._ax.set_title(self._title)
        self._ax.legend()

        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

        fileName = self._output_path + '/plot_' + str(datetime.now())[0:19] + "_" + self._title 
        fileName = fileName.replace("-", "_").replace(":", "_").replace(" ", "_") + '.png'

        print("\n > Save the plot in {}...".format(fileName))
        self._fig.savefig(fileName)
        
    def show():
        plt.show()

##################################################################################################################
############################## > DEEP LEARNING MODEL < ###########################################################
##################################################################################################################

class Network(nn.Module):
    def __init__(self, vectorizer_size : int):
        super().__init__()

        print("\n > Initalizating model...")

        allLayers = [vectorizer_size] + LAYERS
        self._layers = nn.ModuleList(
            nn.Linear(allLayers[i], allLayers[i+1]).to(DEVICE)
            for i in range(len(allLayers)-1)
        )
        self._layers_len = len(self._layers)

        self._dropout = nn.Dropout(DROPOUT).to(DEVICE)
        
        self._loss_criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    def forward(self, inputs : torch.tensor):

        outputs = inputs.float()

        outputs = self._dropout(outputs)

        for i in range(self._layers_len):
            outputs = self._layers[i](outputs)
            if i != self._layers_len-1:
                outputs = F.relu(outputs)

        return outputs.squeeze(1)

    def forwardToPrediction(self, fw : torch.tensor):
        return torch.IntTensor(list(map(lambda x: 0.5 <= x, fw.tolist()))).to(DEVICE)

    def accuracy(self, outputs : torch.tensor, tgts : torch.tensor):
        return torch.sum(torch.eq(self.forwardToPrediction(outputs), tgts))

    def loss(self, outputs : torch.tensor, tgts : torch.tensor):
        return self._loss_criterion(outputs.float(), tgts.float())

    def save(self):
    
        print("\n > Saving model in {}...".format(MODEL_FOLDER_NAME + MODEL_FILE_NAME))
        torch.save(self.state_dict(), os.path.join(MODEL_FOLDER_NAME, MODEL_FILE_NAME))

    def load(self):
        
        print("\n > Loading model...")
                
        if not os.path.exists(MODEL_FOLDER_NAME):
            print("Creating model folder {}...".format(MODEL_FOLDER_NAME))
            os.makedirs(MODEL_FOLDER_NAME)

        if os.path.exists(MODEL_FOLDER_NAME + MODEL_FILE_NAME):
            print("Leading model {}...".format(MODEL_FOLDER_NAME + MODEL_FILE_NAME))
            self.load_state_dict(torch.load(MODEL_FOLDER_NAME + MODEL_FILE_NAME))

##################################################################################################################
############################## > TRAIN DATA < ####################################################################
##################################################################################################################

def _train(model : Network, data_iterator, optim):

    print("\n > Training model...")
    
    total = 0
    loss_total = 0
    accuracy_total = 0
    model.train()
    
    for curr_batch in data_iterator:
        texts, labels = curr_batch

        optim.zero_grad()

        preds = model(texts)
        loss_curr = model.loss(preds, labels)
        accuracy_curr = model.accuracy(preds, labels)
        
        loss_curr.backward()
        optim.step()
        
        total += len(texts)
        loss_total += loss_curr.item()
        accuracy_total += accuracy_curr.item()

    print('Training loss: {:.3f}'.format(loss_total / total))
    print('Training accuracy: {:.3f}%'.format(accuracy_total * 100 / total))
        
    model.save()

    return loss_total / total, accuracy_total * 100 / total

def _eval(model : Network, data_iterator):
    
    print("\n > Evaluating model...")

    total = 0
    loss_total = 0
    accuracy_total = 0
    model.eval()
    
    with torch.no_grad():

        for curr_batch in data_iterator:
            texts, labels = curr_batch

            preds = model(texts)
            loss_curr = model.loss(preds, labels)
            accuracy_curr = model.accuracy(preds, labels)

            total += len(texts)
            loss_total += loss_curr.item()
            accuracy_total += accuracy_curr.item()

    print('Evaluating loss: {:.3f}'.format(loss_total / total))
    print('Evaluating accuracy: {:.3f}%'.format(accuracy_total * 100 / total))
    print()
        
    return loss_total / total, accuracy_total * 100 / total

def learn(model : Network, datasets : Datasets):
    
    print("\n > Learning with model...")

    optim = torch.optim.Adam(model.parameters(), lr=STEP)
    lossPlot = SmartPlot("Loss", "Epochs", "Loss ", PLOTS_FOLDER_NAME)
    accuracyPlot = SmartPlot("Accuracy", "Epochs", "Accuracy ", PLOTS_FOLDER_NAME)

    for ep in tqdm(range(TOTAL_EPOCHS)):

        # Train
        tLoss, tAccuracy = _train(model, datasets.learningLoader, optim)
        lossPlot.addPoint("Training", "red", tLoss)
        accuracyPlot.addPoint("Training", "red", tAccuracy)

        # Evaluate
        eLoss, eAccuracy = _eval(model, datasets.testingLoader)
        lossPlot.addPoint("Evaluation", "green", eLoss)
        accuracyPlot.addPoint("Evaluation", "green", eAccuracy)
    
    lossPlot.build()
    accuracyPlot.build()

    SmartPlot.show()

##################################################################################################################
############################## > PREDICT DATA < ##################################################################
##################################################################################################################

def predict(model, vectorizer : Vectorizer, sentance : str):
    
    print("\n > Predicting with model...")

    model.eval()

    with torch.no_grad():
        X = torch.IntTensor(vectorizer.transform([sentance]).toarray()).to(DEVICE)
        Y = model.forwardToPrediction(model(X))

        print("Sentance \"{s}\" : {p}".format(s=sentance, p=Y.item()))

##################################################################################################################
############################## > FILES < #########################################################################
##################################################################################################################

def rmdir(directory):

    if os.path.exists(directory):
        directory = Path(directory)
        for item in directory.iterdir():
            if item.is_dir():
                rmdir(item)
            else:
                item.unlink()
        directory.rmdir()

##################################################################################################################
############################## > DISABLE PRINT < #################################################################
##################################################################################################################

class HiddenPrints:
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')


    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

##################################################################################################################
############################## > MAIN < ##########################################################################
##################################################################################################################

def main():

    if "clear" in TODO:
        rmdir(VECTORIZER_FOLDER_NAME)
        rmdir(MODEL_FOLDER_NAME)
        rmdir(PLOTS_FOLDER_NAME)

    if "load" in TODO or "train" in TODO or "predict" in TODO:
        dataLoader = DataLoader()

        if "load" in TODO:
            Vectorizer.Load(dataLoader)

        if "train" in TODO or "predict" in TODO:        

            vectorizer, vectorizerSize = Vectorizer.Get()
            datasets = Datasets(dataLoader, vectorizer)
            model = Network(vectorizerSize).to(DEVICE)
            model.load()

            if "train" in TODO:
                learn(model, datasets)

            if "predict" in TODO:
                predict(model, vectorizer, PREDICTION_SENTANCE)


if __name__ == "__main__":

    if WITHOUT_PRINTS:
        with HiddenPrints():
            main()
    else:
        main()