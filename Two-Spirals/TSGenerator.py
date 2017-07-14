
# coding: utf-8

# In[13]:

# Two-Spirals problem datasets generator by CaTheother
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

cols = np.array(['x','y','cls']);
df = pd.DataFrame();
df0 = pd.DataFrame();
df1 = pd.DataFrame();


# In[14]:

for i in range(0,100):
    phi = i * np.pi /32;
    r = 6.5 *(104-i)/104;
    x = r * np.cos(phi);
    y = r * np.sin(phi);
    data0 = np.array([[x,y,0]]);
    data1 = np.array([[-x,-y,1]]);
    df_temp0 = pd.DataFrame(data0, columns = cols);
    df_temp1 = pd.DataFrame(data1, columns = cols);
    df0 = df0.append(df_temp0,ignore_index=True);
    df1 = df1.append(df_temp1,ignore_index=True);
df = df.append(df0,ignore_index=True);
df = df.append(df1,ignore_index=True);
df.to_csv('TSdataset.csv');


# In[15]:

get_ipython().magic('matplotlib inline')
plt.scatter(df0['x'],df0['y']);
plt.scatter(df1['x'],df1['y']);

