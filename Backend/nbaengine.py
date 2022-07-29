import pandas as pd 
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

player = 'college_player-2021.csv'
draft = 'draft-2021.csv'

def clean_data(player, draft):
    college_data = pd.read_csv(player)
    draft_data = pd.read_csv(draft)  

    iloc_index = 0 #used for calling row, subtract 1 if one is dropped
    drop_index = 0 #used for drop indexing, no subtract 1
    length = len(college_data)
    while drop_index<length:
        row = list(college_data.iloc[iloc_index])
        null = pd.isna(row)
        if True in null:
            college_data.drop(drop_index, axis=0, inplace=True)
            iloc_index -= 1
        iloc_index += 1
        drop_index += 1

    drafted = list()
    for x in range(len(college_data)):
        row = college_data.iloc[x]
        if draft_data['Player'].str.contains(row['Player']).any():
            drafted.append(1)
        else:
            drafted.append(0)
    college_data['Drafted'] = drafted
    
    
    college_data = college_data.set_index('Player')
    return college_data

def get_dummies(college_data):
    #dummies
    conference = pd.get_dummies(college_data['Conference'])
    player_class = pd.get_dummies(college_data['Class'])
    position = pd.get_dummies(college_data['Pos'])
    school = pd.get_dummies(college_data['School'])

    college_data = college_data.drop('Conference', axis=1)
    college_data = college_data.drop('Class', axis=1)
    college_data = college_data.drop('Pos', axis=1)
    college_data = college_data.drop('School', axis=1)

    college_data = pd.concat([college_data, conference], axis=1)
    college_data = pd.concat([college_data, player_class], axis=1)
    college_data = pd.concat([college_data, position], axis=1)
    college_data = pd.concat([college_data, school], axis=1)

    college_data = college_data.sample(frac=1).reset_index(drop=True)
    
    return college_data

dummy_data = clean_data(player, draft)
model_data = get_dummies(dummy_data)

def run_model(data):
    X = data.drop('Drafted', axis=1) 
    Y = data['Drafted'] 

    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, Y_over = oversample.fit_resample(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X_over, Y_over, test_size=0.3)

    mlp = MLPClassifier(hidden_layer_sizes=(4,3), activation='relu', max_iter=400)
    mlp.fit(X_train,y_train)

    pred = mlp.predict(X_test)

    cm = confusion_matrix(y_test, pred, labels=mlp.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
    disp.plot()
    plt.show()

run_model(model_data)

def run_engine(player_data, draft_data):
    dummy_data = clean_data(player_data, draft_data)
    model_data = get_dummies(dummy_data)
    

