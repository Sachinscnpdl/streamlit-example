from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

#Import Libraries
from pymatgen.core.composition import Composition, Element
from pymatgen.core.structure import SiteCollection
from matminer.featurizers.composition.alloy import Miedema, WenAlloys,YangSolidSolution
from matminer.featurizers.composition import ElementFraction
from matminer.featurizers.conversions import StrToComposition
from matminer.utils.data import MixingEnthalpy, DemlData
from matminer.utils import data_files #for importing "Miedema.csv" present inside package of Matminer library
from matplotlib.ticker import MultipleLocator # for minor tick lines
import seaborn as sns

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os
import matplotlib.pyplot as plt

ef= ElementFraction()
stc = StrToComposition()

# Define a list of chemical formulas
chemical_formulas = ['H2O', 'CO2', 'NaCl']

# Set Streamlit app title
st.title('Chemical Formula Selection')

# Add a dropdown to select a chemical formula
selected_formula = st.selectbox('Select a chemical formula', chemical_formulas)

# Display the selected formula
st.write('Selected formula:', selected_formula)



"""

# Welcome to PiezoTensorNet!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""
#############################################################################################################################
# Add the prediction files
df_piezo = pd.read_csv('csv/For_Prediction.csv')
df_piezo = df_piezo.head(50)

df_piezo = stc.featurize_dataframe(df_piezo, "formula_pretty",ignore_errors=True,return_errors=True)
df_piezo = ef.featurize_dataframe(df_piezo, "composition",ignore_errors=True,return_errors=True)

# In[4]:
from matminer.featurizers.composition import ElementProperty
featurizer = ElementProperty.from_preset('magpie')
df_piezo = featurizer.featurize_dataframe(df_piezo, col_id='composition')
#y = bg_data_featurized['gap expt']

# In[5]:
get_ipython().run_line_magic('run', 'functions.ipynb')
df, df_input_target = properties_calculation(df_piezo)

# In[6]:
magpie_list = ['MagpieData minimum Number',
 'MagpieData maximum Number',
 'MagpieData range Number',
 'MagpieData mean Number',
 'MagpieData avg_dev Number',
 'MagpieData mode Number',
 'MagpieData minimum MendeleevNumber',
 'MagpieData maximum MendeleevNumber',
 'MagpieData range MendeleevNumber',
 'MagpieData mean MendeleevNumber',
 'MagpieData avg_dev MendeleevNumber',
 'MagpieData mode MendeleevNumber',
 'MagpieData minimum AtomicWeight',
 'MagpieData maximum AtomicWeight',
 'MagpieData range AtomicWeight',
 'MagpieData mean AtomicWeight',
 'MagpieData avg_dev AtomicWeight',
 'MagpieData mode AtomicWeight',
 'MagpieData minimum MeltingT',
 'MagpieData maximum MeltingT',
 'MagpieData range MeltingT',
 'MagpieData mean MeltingT',
 'MagpieData avg_dev MeltingT',
 'MagpieData mode MeltingT',
 'MagpieData minimum Column',
 'MagpieData maximum Column',
 'MagpieData range Column',
 'MagpieData mean Column',
 'MagpieData avg_dev Column',
 'MagpieData mode Column',
 'MagpieData minimum Row',
 'MagpieData maximum Row',
 'MagpieData range Row',
 'MagpieData mean Row',
 'MagpieData avg_dev Row',
 'MagpieData mode Row',
 'MagpieData minimum CovalentRadius',
 'MagpieData maximum CovalentRadius',
 'MagpieData range CovalentRadius',
 'MagpieData mean CovalentRadius',
 'MagpieData avg_dev CovalentRadius',
 'MagpieData mode CovalentRadius',
 'MagpieData minimum Electronegativity',
 'MagpieData maximum Electronegativity',
 'MagpieData range Electronegativity',
 'MagpieData mean Electronegativity',
 'MagpieData avg_dev Electronegativity',
 'MagpieData mode Electronegativity',
 'MagpieData minimum NsValence',
 'MagpieData maximum NsValence',
 'MagpieData range NsValence',
 'MagpieData mean NsValence',
 'MagpieData avg_dev NsValence',
 'MagpieData mode NsValence',
 'MagpieData minimum NpValence',
 'MagpieData maximum NpValence',
 'MagpieData range NpValence',
 'MagpieData mean NpValence',
 'MagpieData avg_dev NpValence',
 'MagpieData mode NpValence',
 'MagpieData minimum NdValence',
 'MagpieData maximum NdValence',
 'MagpieData range NdValence',
 'MagpieData mean NdValence',
 'MagpieData avg_dev NdValence',
 'MagpieData mode NdValence',
 'MagpieData minimum NfValence',
 'MagpieData maximum NfValence',
 'MagpieData range NfValence',
 'MagpieData mean NfValence',
 'MagpieData avg_dev NfValence',
 'MagpieData mode NfValence',
 'MagpieData minimum NValence',
 'MagpieData maximum NValence',
 'MagpieData range NValence',
 'MagpieData mean NValence',
 'MagpieData avg_dev NValence',
 'MagpieData mode NValence',
 'MagpieData minimum NsUnfilled',
 'MagpieData maximum NsUnfilled',
 'MagpieData range NsUnfilled',
 'MagpieData mean NsUnfilled',
 'MagpieData avg_dev NsUnfilled',
 'MagpieData mode NsUnfilled',
 'MagpieData minimum NpUnfilled',
 'MagpieData maximum NpUnfilled',
 'MagpieData range NpUnfilled',
 'MagpieData mean NpUnfilled',
 'MagpieData avg_dev NpUnfilled',
 'MagpieData mode NpUnfilled',
 'MagpieData maximum NdUnfilled',
 'MagpieData range NdUnfilled',
 'MagpieData mean NdUnfilled',
 'MagpieData avg_dev NdUnfilled',
 'MagpieData mode NdUnfilled',
 'MagpieData maximum NfUnfilled',
 'MagpieData range NfUnfilled',
 'MagpieData mean NfUnfilled',
 'MagpieData avg_dev NfUnfilled',
 'MagpieData minimum NUnfilled',
 'MagpieData maximum NUnfilled',
 'MagpieData range NUnfilled',
 'MagpieData mean NUnfilled',
 'MagpieData avg_dev NUnfilled',
 'MagpieData mode NUnfilled',
 'MagpieData minimum GSvolume_pa',
 'MagpieData maximum GSvolume_pa',
 'MagpieData range GSvolume_pa',
 'MagpieData mean GSvolume_pa',
 'MagpieData avg_dev GSvolume_pa',
 'MagpieData mode GSvolume_pa',
 'MagpieData minimum GSbandgap',
 'MagpieData maximum GSbandgap',
 'MagpieData range GSbandgap',
 'MagpieData mean GSbandgap',
 'MagpieData avg_dev GSbandgap',
 'MagpieData mode GSbandgap',
 'MagpieData maximum GSmagmom',
 'MagpieData range GSmagmom',
 'MagpieData mean GSmagmom',
 'MagpieData avg_dev GSmagmom',
 'MagpieData mode GSmagmom',
 'MagpieData minimum SpaceGroupNumber',
 'MagpieData maximum SpaceGroupNumber',
 'MagpieData range SpaceGroupNumber',
 'MagpieData mean SpaceGroupNumber',
 'MagpieData avg_dev SpaceGroupNumber',
 'MagpieData mode SpaceGroupNumber']


# In[7]:
#df_fs_magpie = df.iloc[:,list(range(92,220))]
# df_fs_magpie = df.iloc[:,list(range(85,213))]
df_fs_magpie =df.loc[:, magpie_list]

# In[8]:
df_input_target = df_input_target.drop(['No of Components'], axis=1)

# In[9]:
df_input_target= df_input_target.iloc[:,list(range(0,13))]

# In[10]:
df_features = pd.concat([df_fs_magpie,df_input_target], axis=1)

# # Classification Predictions

# In[11]:
path='model_files//nn_model//classification//'

# In[12]:
import pickle
scaler = pickle.load(open(path+'scaler.pkl','rb'))

df_std = scaler.transform(df_features)

pca_1 = pickle.load(open(path+'pca_1.pkl','rb'))
#std_test = scaler.transform(X_test_std)
#test_pca_1 =  pca_1.transform(std_test)

df_pca =  pca_1.transform(df_std)

# In[13]:
from tensorflow import keras
model_cat = keras.models.load_model('model_files/nn_model/classification/model_cat.h5')
model_cata = keras.models.load_model('model_files/nn_model/classification/model_cata.h5')
model_catb = keras.models.load_model('model_files/nn_model/classification/model_catb.h5')

y_cat = model_cat.predict(df_pca)

# In[14]:
category = np.where(y_cat[:, 0] > 0.5, 'A', 'B')
subcategories = []
y_tensor = []

if np.any(category == 'A'):
    y_subcat = model_cata.predict(df_pca)
    
    for subcat in y_subcat:
        subcategory = []
        y_target = []
        if subcat[0] > 0.33:
            subcategory.append('cubic')
#             y_value = ensemble_model(model_path='model_files/nn_model/cubic/')
            
        elif subcat[1] > 0.33:
            subcategory.append('tetra42m')
#             y_value = ensemble_model(model_path='model_files/nn_model/tetra42m/')
            
        elif subcat[2] > 0.33:
            subcategory.append('ortho222')
#             y_value = ensemble_model(model_path='model_files/nn_model/ortho222/')
            
        subcategories.append(subcategory)
#         y_tensor.append(y_value)
        
else:
    y_subcat = model_catb.predict(df_pca)
    for subcat in y_subcat:
        subcategory = []
        y_target = []
        
        if subcat[0] > 0.5:
            subcategory.append('orthomm2')
#             y_value = ensemble_model(model_path='model_files/nn_model/orthomm2/')
            
        elif subcat[1] > 0.5:
            subcategory.append('hextetramm')
#             y_value = ensemble_model(model_path='model_files/nn_model/hextetramm/')
            
        subcategories.append(subcategory)
#         y_tensor.append(y_value)


# In[15]:

import multiprocessing
import os
import numpy as np
import pickle
import keras

# Define the number of processes to use
num_processes = multiprocessing.cpu_count()

# Create a shared dictionary to cache models
manager = multiprocessing.Manager()
model_cache = manager.dict()

def ensemble_model(df_pred, model_path= 'model_files/nn_model/cubic/'):
    
    # Assuming your data is stored in 'data' variable
    df_pred = df_pred.reshape(1, -1)
    # Check if the model is already in the cache
    if model_path in model_cache:
        return model_cache[model_path]

    scaler = pickle.load(open(model_path+'scaler_reg.pkl', 'rb'))
    df_std = scaler.transform(df_pred)

    pca_1 = pickle.load(open(model_path+'pca_reg.pkl', 'rb'))
    df_pca = pca_1.transform(df_std)

    model1 = keras.models.load_model(model_path+'model_1.h5')
    model2 = keras.models.load_model(model_path+'model_2.h5')
    model3 = keras.models.load_model(model_path+'model_3.h5')
    model4 = keras.models.load_model(model_path+'model_4.h5')
    model5 = keras.models.load_model(model_path+'model_5.h5')

    predictions = []

    for model in [model1, model2, model3, model4, model5]:
        pred = model.predict(df_pca)  # Assuming the models have a predict() method
        predictions.append(pred)

    ensemble_prediction = np.mean(predictions, axis=0)  # Average the predicted probabilities across models
    ensemble_prediction = ensemble_prediction.tolist()
    ensemble_prediction = ensemble_prediction[0]

    # Store the prediction in the cache
    model_cache[model_path] = ensemble_prediction
    return ensemble_prediction

# In[16]:
df_pca = df_features.values
ensemble_model(df_pca[5], model_path='model_files/nn_model/hextetramm/')

# In[17]:
print("This has to be printed")
df_predict = df_features.values
y_tensor = []
y_value = []
# subcategories =  []
for item in range(df_pca.shape[0]):
#     print(item, y_cat[item], subcategories[item])
    if y_cat[item][0] > 0.5 and subcategories[item] == ['cubic']:
#         if subcategories[item] == 'cubic':
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/cubic/')
        y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[0], 0], [0, 0, 0, 0, 0, y[0]]]
            
    elif y_cat[item][0] > 0.5 and subcategories[item] == ['tetra42m']:
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/tetra42m/') 
        y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[0], 0], [0, 0, 0, 0, 0, y[1]]]
            
    elif y_cat[item][0] > 0.5 and subcategories[item] == ['ortho222']:
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/ortho222/') 
        y_value = [[0, 0, 0, y[0], 0, 0], [0, 0, 0, 0, y[1], 0], [0, 0, 0, 0, 0, y[2]]]
            
    elif y_cat[item][0] < 0.5 and subcategories[item] == ['orthomm2']:
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/orthomm2/')
        y_value = [[0, 0, 0, 0, y[0], 0], [0, 0, 0, y[1], 0, 0], [y[2], y[3], y[4], 0, 0, 0]]
            
    elif y_cat[item][0] < 0.5 and subcategories[item] == ['hextetramm']:
        y = ensemble_model(df_predict[item], model_path='model_files/nn_model/hextetramm/') 
        y_value = [[0, 0, 0, 0, y[0], 0], [0, 0, 0, y[0], 0, 0], [y[1], y[2], y[3], 0, 0, 0]]

    y_tensor.append(y_value)

# In[18]:
y_tensor

# In[19]:
y_tensor[1]
##################################################################################################
# End of Prediction



with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
