import streamlit as st
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle


# ---- Laptop Price Predictor Home Page ----
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# import the model & DataFrame
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Setting Up Title :
page_title = "Laptop Price Predictor"
text_style = f"<h1 style='font-family:Georgia, serif; text-align:center; color:#fada5e; font-size:35px;'>{page_title}</h1>"
st.markdown(text_style, unsafe_allow_html=True)

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
# Encoding Ram
if ram == 2:
    ram = 0
elif ram == 4:
    ram = 1
elif ram == 6:
    ram = 2
elif ram == 8:
    ram = 3
elif ram == 12:
    ram = 4
elif ram == 16:
    ram = 5
elif ram == 24:
    ram = 6
elif ram == 32:
    ram = 7
else:
    ram = 8

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
# Encoding Touchscreen
if touchscreen == 'No':
    touchscreen = 0
else:
    touchscreen = 1

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])
# Encoding IPS
if ips == 'No':
    ips = 0
else:
    ips = 1

# resolution
resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                           '2560x1440', '2304x1440']
                          )

# CPU Brand
cpu = st.selectbox('CPU', df['CPU_Name'].unique())
# Encoding CPU
if cpu == 'Intel Core i3':
    cpu = 0
elif cpu == 'Other Intel Processor':
    cpu = 1
elif cpu == 'AMD Processor':
    cpu = 2
elif cpu == 'Intel Core i5':
    cpu = 3
else:
    cpu = 4

# HDD Storage
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD Storage
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU Brand
gpu = st.selectbox('GPU', df['GPU_Brand'].unique())
# Encoding GPU_Brand
if gpu == 'AMD':
    gpu = 0
elif gpu == 'Intel':
    gpu = 1
else:
    gpu = 2

# OS Type
os = st.selectbox('OS', df['OS'].unique())
# Encoding OS
if os == 'Others/No OS/Linux':
    os = 0
elif os == 'Windows':
    os = 1
else:
    os = 2

if st.button('Predict Price'):
    query = np.array([ram, touchscreen, ips, cpu, gpu, os, company, type, weight, resolution, hdd, ssd]).reshape(1, 12)
    result = f"The predicted price of this configuration is ₹ {format(int(np.exp(pipe.predict(query)[0])), ',d')}"
    text_style = f"<h1 style='font-family:Georgia, serif; text-align:center; color:#fada5e; font-size:20px;'>{result}</h1>"
    st.markdown(text_style, unsafe_allow_html=True)


for i in range(5):
    st.write('')

with st.expander("About Prediction"):
    st.write('Uses RANDOM FOREST model to predict the Laptop price.')
    st.write('Model has a R2_Score of 0.86 & MAE of 0.16.')

with st.expander("KNOW MORE : Influential Factors"):
    col1, col2 = st.columns(2)
    # Company Vs Price
    with col1:
        st.write('Company Vs Price')
        fig = plt.figure()
        sns.barplot(data=df, x='Price', y='Company')
        st.pyplot(fig)
    # Ram Vs Price
    with col2:
        st.write('RAM Vs Price')
        fig = plt.figure()
        sns.barplot(x=df['Ram'], y=df['Price'])
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    # OS Vs Price
    with col1:
        st.write('OS Vs Price')
        fig = plt.figure()
        sns.barplot(x=df['Price'], y=df['OS'])
        st.pyplot(fig)
    # CPU Vs Price
    with col2:
        st.write('CPU Vs Price')
        fig = plt.figure()
        sns.barplot(x=df['Price'], y=df['CPU_Name'])
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    # Touch_Screen Vs Price
    with col1:
        st.write('Touch Screen Vs Price')
        fig = plt.figure()
        sns.barplot(x=df['Touchscreen'], y=df['Price'])
        st.pyplot(fig)
    # IPS_Display Vs Price
    with col2:
        st.write('IPS Display Vs Price')
        fig = plt.figure()
        sns.barplot(x=df['IPS Display'], y=df['Price'])
        st.pyplot(fig)
