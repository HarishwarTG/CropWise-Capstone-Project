from sklearn.preprocessing import StandardScaler , LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def preprocess(data_path):
    df = pd.read_csv(data_path)
    X=df.drop('label',axis=1)
    y=df['label']
    ss=StandardScaler()
    le=LabelEncoder()
    X_std = ss.fit_transform(X)
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_std,y_encoded,test_size=0.3,stratify=y,random_state=42)
    
    with open('artifacts/ss_model.pkl', 'wb') as f:
        pickle.dump(ss, f)

    with open('artifacts/le_model.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    return X_train, X_test, y_train, y_test

def build_model(data_path,model_path):
    
    X_train, X_test, y_train, y_test = preprocess(data_path)
    
    ##convert labels to OHE
    y_train_ohe = to_categorical(y_train)
    y_test_ohe = to_categorical(y_test)
    
    ## INtialize model
    model= Sequential()
    # 1. add input layer --> 7 features
    model.add(Dense(units=64, input_shape=(X_train.shape[1],)))
    model.add(LeakyReLU(alpha=0.1))
    
    #2.Hidden layers
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    
    model.add(Dense(units=64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=32))
    model.add(LeakyReLU(alpha=0.1))
    
    #3.Output layer --> 22 crops and softmax for multiclass
    model.add(Dense(units=22, activation='softmax'))
    
    #4. compile and fit the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['Accuracy'])
    earlystop = EarlyStopping(monitor='val_loss' , mode ='min' , verbose =1 , patience=25)
    model.fit(X_train, y_train_ohe, validation_split=0.3 , epochs = 100 , batch_size=20, callbacks=[earlystop]) 
    
    #5. predict the data
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred , axis =1) # as it will give probability of all the crop , we have to select max value as output
    
    acc = accuracy_score(y_test , y_pred)
    
    print('accuracy:',acc)
    
    model.save(model_path)
    

def main():
    data_path = "artifacts/Optimizing Agriculture Production.csv"
    model_path = "artifacts/crop_nw_model.h5"
    build_model(data_path,model_path)
    

if __name__ == "__main__":
    main()
    