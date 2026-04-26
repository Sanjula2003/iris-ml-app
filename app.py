import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("KNN-iris-model.sav")

flower_images = {
    "Iris-setosa": "setosa.jpg",
    "Iris-versicolor": "versicolor.jpg",
    "Iris-virginica": "virginica.jpg"
}

# Page config
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS Styling
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #fdfbfb, #ebedee);
}

.title {
    text-align: center;
    font-size: 40px;
    color: #6a0dad;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 20px;
}

.result-box {
    background-color: #f0f8ff;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    color: #2c3e50;
    font-weight: bold;
    border: 2px solid #dcdcdc;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    background-color: #6a0dad;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #8a2be2;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">🌸 Iris Flower Classification</p>', unsafe_allow_html=True)


# Image
#st.image("iris.jpeg", use_container_width=True)

# Sidebar
st.sidebar.header("Flower Measurements")
st.sidebar.write("Adjust the sliders to predict flower species")

# Sliders in Sidebar
sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 3.0, 1.2)

# Display input values in columns
st.subheader("Input Features")

col1, col2 = st.columns(2)

with col1:
    st.metric("Sepal Length", f"{sepal_length} cm")
    st.metric("Petal Length", f"{petal_length} cm")

with col2:
    st.metric("Sepal Width", f"{sepal_width} cm")
    st.metric("Petal Width", f"{petal_width} cm")

# Prediction Button
if st.button("Predict Flower Type"):

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(input_data)
    result = prediction[0]

    st.markdown(
        f'''<div class="result-box">🌼 Predicted Flower: {result}</div>''',
        unsafe_allow_html=True
    )

    # Show corresponding image
    if result in flower_images:
        st.image(flower_images[result], caption=result, use_container_width=True)

    st.balloons()








