import streamlit as st
from PIL import Image
import tempfile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import cv2
import numpy as np
from matplotlib import pyplot as plt

gauth = GoogleAuth()
drive = GoogleDrive(gauth)

#folder id: 1xOM8rrfKVyc3csfra75NkeJ2LDEElknu, can be extracted from url of folder
folder = '1xOM8rrfKVyc3csfra75NkeJ2LDEElknu'

st.set_page_config(
 page_icon="💧",
 layout="wide",
 page_title = "Image Processing"
)

col1, col2, col3= st.columns(3, gap = "medium")

def get_image_data():
    # Download first image file from specified folder in Google Drive into memory
    file_list = drive.ListFile({'q': f"'{folder}' in parents and trashed=false and mimeType contains 'image/'"}).GetList()
    if file_list:
        file = file_list[0]
        print('file downloaded : ', file['title'])
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.GetContentFile(temp.name)
            return temp.name
    else:
        return None
    
with col2:
    st.title(':blue[IMAGE PROCESSING]')
    col3, col4, col5= st.columns(3, gap = "small")
    with col4:
        if st.button('Refresh Image'):
            get_image_data()


 # Download image data from Google Drive
image_file = get_image_data()


col1, col2= st.columns(2, gap = "large")

if image_file:

    # Open image from local system using PIL
    image = Image.open(image_file)

    with col1:   
        #Original Image
        st.image(image, width=600, caption='Original Image captured by Buoy')
        
        
    with col2:
        # Convert PIL Image to OpenCV Image
        open_cv_image = np.array(image) 
        
        # Check if image is grayscale
        if len(open_cv_image.shape) == 3:
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
        
        # Add your image processing code here
        # read the image into a numpy array using OpenCV
        # perform image processing operations here...
        processed_img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # Apply median filter to remove noise
        median = cv2.medianBlur(processed_img, 5)

        # Apply adaptive thresholding to segment the water region
        thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Calculate the mean intensity of the water region
        mean_intensity = np.mean(median[np.where(thresh ==0)])

        # Normalize the mean intensity by the maximum intensity value
        turbidity = mean_intensity / 10.0
        

        # Load the image
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)

        if len(img.shape) == 3:
            # Convert the image to LAB color space
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            # Extract the green channel
            green_channel = img_lab[:,:,1]

            # Calculate the mean green value
            mean_green = np.mean(green_channel)

            # Calculate the chlorophyll level
            chlorophyll = 12.7 * mean_green - 8.06
            
        # st.markdown('Processed Image: ')

        # Display processed image using Streamlit
        st.image(processed_img, width=600, caption='Processed Image')

        

    col11, col13, col14 = st.columns(3, gap = "large")
    res = ":orange[Turbidity] = " + str(round(turbidity,3)) + "ntu"
    res1 = " :orange[Chlorophyll] = " + str(round(chlorophyll, 3)) + "mg/L"

    with col13:
        st.subheader(res)
        st.subheader(res1)
        
else:
    col21, col23, col24 = st.columns([1,2,1], gap = "medium")
    with col23:
        st.subheader(' :red[NO IMAGES FOUND IN SPECIFIED FOLDER..............!] ')
        



