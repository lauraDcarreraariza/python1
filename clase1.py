import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
np.random.seed(1234)
st.title('RShelp')
datos=np.random.normal(0,1, size=(100,4))
data=pd.DataFrame(datos,
                  columns=list('ABCD'))
st.dataframe(data)
e=np.random.normal(0,1, size =100)
y=data['A']*2 + data['B']*3+ data['C']*4+ data['D']*0.3 +10 +e
model = DecisionTreeRegressor(max_depth=4)
model.fit(data,y)
st.subheader('A')
val_a=st.slider('Seleccione el valor de A',
          data['A'].min(),
          data['A'].max())
st.subheader('B')
val_b=st.slider('Seleccione el valor de B',
          data['B'].min(),
          data['B'].max())
st.subheader('C')
val_c=st.slider('Seleccione el valor de C',
          data['C'].min(),
          data['C'].max())
st.subheader('D')
val_d=st.slider('Seleccione el valor de D',
          data['D'].min(),
          data['D'].max())
valores=np.array([[val_a,val_b,val_c,val_d]])
pre=model.predict(valores)
st.write(pre)