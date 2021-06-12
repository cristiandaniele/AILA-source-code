import pandas as pd
import numpy as np
from numpy import nan
def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele+" "
    # return string
    return str1
filepath_dict = {'myStandard':   './dataset.txt'}
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)
df.dropna()
df["label"] = pd.to_numeric(df["label"],errors='raise')
df['sentence'] = df['sentence'].astype('str')

from sklearn.model_selection import train_test_split

df_yelp = df[df['source'] == 'myStandard']

sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.25, random_state=1000)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

print("Press enter to train the model on " + str(len(sentences_train))+" records")
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
from keras.backend import clear_session
clear_session()
from keras.models import Sequential
from keras import layers
import os
import tensorflow as tf
from keras.utils.vis_utils import plot_model

input_dim = X_train.shape[1]  # Number of features
print(input_dim)
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu',name='layer1'))
model.add(layers.Dense(1, activation='sigmoid',name='layer2'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
from keras.backend import clear_session
clear_session()

#model.load_weights(checkpoint_path)
# Train the model with the new callback
from keras import backend as K
K.set_value(model.optimizer.learning_rate, 0.0001)
history = model.fit(X_train, y_train,
                    epochs=15,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=50)
import pandas as pd
import numpy as np
import json
policyName="tesla"
with open("./entities_"+policyName+".json", 'r') as in_f:
        data = json.load(in_f)
def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] <= threshold
    return ['background-color: red' if is_max.any() else '' for v in is_max]
metricsEntity=np.zeros((len(data),1))
i=0
mean=[]
os.mkdir("./"+policyName)
for object in data:
  nameFile=(object["entity"]+".html").replace(" ","-")
  #for each entity
  f=open("./"+policyName+"/"+nameFile,"w")
  sentences_test=np.array(object["sentences"])
  y_test=np.zeros((len(sentences_test),1))
  X_test  = vectorizer.transform(sentences_test)
  predictions=model.predict(X_test)
  newPred=[]

  for x in predictions:
    newPred.append(round(x[0],2))
  metricsEntity[i]=0
  for x in newPred:
    #for each entity prediction
    metricsEntity[i]=metricsEntity[i]+x
  mean.append(metricsEntity[i]/len(predictions))
  d = {object["entity"]: list(sentences_test), 'Label': list(newPred)}
  df = pd.DataFrame(data=d)
  s=df.style.apply(highlight_greaterthan, threshold=0.4, column=['Label'], axis=1)
  f.write(s.render())
  f.close()
  i=i+1
j=0
file_index=open("./"+policyName+"/index.html","w")
html="<!DOCTYPE html>\
<html>\
<body>\
<ul>"
i=0
input("Enities extracted: "+str(len(mean)))
toPrint=[]
for value in range (0,len(mean)):
  nameFile=(data[j]["entity"]+".html").replace(" ","-")
  x=round(mean[i][0],2)
  if (x<=0.40):
    color="red"
  if (x>0.40 and x<0.60):
    color="orange"
  if (x>=0.60):
    color="green"
  v=round(1-x,1)
  likelihood=v
  if (v <= 0.2):
    likelihood="VL"
  if (v>0.2 and v<=0.4):
      likelihood="L"
  if (v>0.4 and v<=0.6):
    likelihood="M"
  if (v>0.6 and v<=0.8):
    likelihood="H"
  if (v>0.8):
    likelihood="VH"
  html=html+"<li><a style=\"color:"+color+"\" href=./"+nameFile+">"+(data[j]["entity"])+"<ul><li>Fairness mean: "+str(x)+"</li><li>Likelihood: "+str(likelihood)+"</li></ul></</a></li>"
  # print(data[j]["entity"]+" "+str(round(x,2))+"% negativi")
  toPrint.append( data[j]["entity"] +" : "+ str(x))
  j+=1
  i=i+1
html=html+"</ul>\
</body>\
</html>"
file_index.write(html)
file_index.close()
print("Finished")
