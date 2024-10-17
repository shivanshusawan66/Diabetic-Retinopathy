import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np

# Cache the model loading process
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/trained_model.keras')
    return model

# TensorFlow model prediction
def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                }
        </style>
        """, unsafe_allow_html=True)



app_mode = option_menu(
    menu_title = None,
    options = ["Home", "About Us", "Diagnosis"],
    icons=["house","info-circle","heart"],
    orientation = "horizontal",
    default_index=2
)


# Homepage
if(app_mode == "Home"):
    st.title("Welcome to Diabetic Retinopathy Detection")
    
    # Display an image of diabetic retinopathy
    # st.image("path_to_your_image/diabetic_retinopathy.jpg", use_column_width=True)  # Update the image path

    st.markdown("""
    Diabetic retinopathy is a serious eye condition that affects individuals with diabetes, potentially leading to vision loss. Our application leverages cutting-edge machine learning techniques to **detect diabetic retinopathy early**, enabling timely intervention and effective management.
    """)

    # Display the image
    st.image("img/stageDR.jpg", caption="Stages of Diabetic Retinopathy", use_column_width=True)


    st.markdown("""
    ## How It Works
    1. **Image Upload**: Use the file uploader to upload a retinal image.
    2. **Prediction**: Click the 'Predict' button to start the analysis.
    3. **Results**: Receive instant feedback on the presence and severity of diabetic retinopathy.

    ## Why Early Detection Matters
    Early diagnosis of diabetic retinopathy can significantly reduce the risk of severe vision impairment. Regular screenings and prompt interventions can help manage the disease effectively and protect your vision.
    Start your journey towards better eye health today! 

    ### Ready to Get Started?
    - **Upload your retinal image now** and see how our tool can help you!
    """)

    # Additional Call to Action with a friendly reminder
    st.markdown("""

    🥳 **Let’s begin your journey to healthier eyes!** \n
    If you're ready to analyze your retinal image, head over to the **`Diagnosis`** page for predictions!
    """)

    

# About Us Page 
elif app_mode == "About Us":
    st.header("About Us")
    st.markdown(
        """
        We are a dedicated team of passionate learners working on an innovative solution in the field of **Diabetic Retinopathy Detection**. Our project aims to harness the power of machine learning to aid in the early detection of diabetic retinopathy, enhancing patient outcomes and revolutionizing eye care.

        ### Meet the Team:

        - **Shivanshu Sawan**  
          A final-year student at UIET, Panjab University, with a strong foundation in Computer Science and a keen interest in artificial intelligence and its applications in healthcare. 
          - 🔗 [GitHub](https://github.com/shivanshusawan)
          - 🔗 [LinkedIn](https://www.linkedin.com/in/shivanshu-sawan)

        - **Kshitij Negi**  
          A talented peer who brings creativity and technical prowess to our project. Kshitij's skills in software development and data analysis have been instrumental in our progress.

        - **Zul Quarnain Azam**  
          An enthusiastic developer with a knack for problem-solving, Zul's contributions in research and coding have significantly shaped our project's direction.

        ### Our Vision:
        We believe that technology can transform the healthcare landscape. By combining our expertise, we aim to create a user-friendly tool that can assist medical professionals in diagnosing diabetic retinopathy with accuracy and efficiency.

        ### Connect With Us!
        We're excited to share our journey and findings with you. Stay tuned for updates as we work towards our goal. Your feedback and support mean a lot to us!

        Thank you for visiting our project page! 😊
        """
    )



# Prediction Page
elif app_mode == "Diagnosis":
    st.header("Diabetic Retinopathy Detection")
    
    # Section for downloading the test dataset
    st.markdown("### Download the Test Dataset")
    st.write("""
    Enhance your experience with the Diabetic Retinopathy Detection tool by downloading our comprehensive test dataset. 
    This dataset contains a variety of retinal images, useful for practicing and testing the diagnostic capabilities 
    of your models. Click the button below to download the dataset in `.rar` format.
    """)
    
    # Download button for the test dataset
    with open("test_dataset.rar", "rb") as f:
        bytes_data = f.read()
    
    st.download_button(label="Download Test Dataset",
                       data=bytes_data,
                       file_name="test_dataset.rar",
                       mime="application/octet-stream",
                       help="Click here to download the test dataset")

    st.markdown("---")  # Add a horizontal line for better separation

    # Image uploader with size limit (in bytes)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"], 
                                   help="Upload an image file (max size: 2MB)", 
                                   label_visibility="collapsed")


    # Automatically display the uploaded image
    if test_image is not None:
        # Check the file size
        if test_image.size > 2 * 1024 * 1024:  # 2MB limit
            st.error("File size exceeds 2MB. Please upload a smaller image.")
            test_image = None
        else:
            # Display the image 
            st.image(test_image, caption='Original Image', use_column_width=0.5)


            # Make prediction using the AI model
            if st.button("Predict", help="Make a prediction on the processed image"):
                with st.spinner("Processing... Please wait..."):                 
                    st.write("Our Prediction:")
                    result_index = model_prediction(test_image)

                    # Define class names
                    class_name = ['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']
                    st.success(f"The model predicts it's a: **{class_name[result_index]}**")

                    

    else:
        st.info("✨ **Please upload your retinal image for prediction.**  \n"
                 "To get started, simply select an image from your device.")

