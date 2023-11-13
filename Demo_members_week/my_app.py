''' Feedback buttons
            st.title("The model was...")
            right_button = st.button("Right")
            wrong_button = st.button("Wrong")

            if right_button:
                st.write("Thank you for your feedback!")
                st.balloons()

            if wrong_button:
                st.write("Sorry, we'll try to improve!")
                st.markdown("😢" * 100)'''

# Import required libraries
import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np
from PIL import Image


model_name = "RandomForestClassifier"  # El nombre del modelo que guardaste anteriormente
file_path = os.path.join("models", f'{model_name}.pkl')

def createDataframe():
    # Datos
    Hora_Ini = 109

    # Trip 1: Aldapeta-Eskuzaitzeta
    Hora_Ini_1 = Hora_Ini

    drive_tt_1 = 9.219617
    pt_tt_1 = 35.678
    drive_tc_1 = 16580.6871
    pt_tc_1 = 0.851818
    distance_1 = 9085.308

    # Trip 2: Beasain-Tolosa
    Hora_Ini_2 = Hora_Ini

    drive_tt_2 = 10.234167
    pt_tt_2 = 33.66
    drive_tc_2 = 27812.388625
    # drive_tc_2 = 60000
    pt_tc_2 = 0.998864
    # pt_tc_2 = 0.1
    distance_2 = 15239.665
    # distance_2 = 10000

    # Trip 3: Bergara-Zarautz
    Hora_Ini_3 = Hora_Ini

    drive_tt_3 = 26.1228
    pt_tt_3 = 43.765
    drive_tc_3 = 64859.7043	
    pt_tc_3 = 2.352727
    distance_3 = 35539.564

    trips_demo = pd.DataFrame(columns=['Hora_Ini', 'Per_hog', 'Turismos', 'Edad', 'crnt_tur','drive_tt','pt_tt','drive_tc','pt_tc','distance','Tipo_familia'])
    fila1 = pd.Series({'Hora_Ini': Hora_Ini_1, 'Per_hog': 0, 'Turismos': 0, 'Edad': 0, 'crnt_tur': 0,'drive_tt': drive_tt_1,'pt_tt': pt_tt_1,'drive_tc': drive_tc_1,'pt_tc': pt_tc_1,'distance': distance_1, 'Tipo_familia': 0})
    fila2 = pd.Series({'Hora_Ini': Hora_Ini_2, 'Per_hog': 0, 'Turismos': 0, 'Edad': 0, 'crnt_tur': 0,'drive_tt': drive_tt_2,'pt_tt': pt_tt_2,'drive_tc': drive_tc_2,'pt_tc': pt_tc_2,'distance': distance_2, 'Tipo_familia': 0})
    fila3 = pd.Series({'Hora_Ini': Hora_Ini_3, 'Per_hog': 0, 'Turismos': 0, 'Edad': 0, 'crnt_tur': 0,'drive_tt': drive_tt_3,'pt_tt': pt_tt_3,'drive_tc': drive_tc_3,'pt_tc': pt_tc_3,'distance': distance_3, 'Tipo_familia': 0})
    trips_demo = pd.concat([trips_demo, pd.DataFrame([fila1]), pd.DataFrame([fila2]), pd.DataFrame([fila3])], ignore_index=True)
    return trips_demo


def predict_demo(Per_hog, Turismos, Edad, crnt_tur, Tipo_familia, model, trips_demo):
    
    a = {"1-2": 1, "3-5": 2, "6+": 3}
    b = {"0": 1, "1": 2, "2": 3,  "3+": 4}
    c = {"<19": 2, "20-29": 3, "30-44": 4,  "45-59": 5, "60-74": 6, "75+": 7}
    d = {"Yes": 1, "No": 2}
    e = {"Single": 1, "Single parent": 5, "Couple": 3,  "Couple with kids": 4}

    trips_demo = trips_demo.assign(Per_hog = a[Per_hog])
    trips_demo = trips_demo.assign(Turismos = b[Turismos])
    trips_demo = trips_demo.assign(Edad = c[Edad])
    trips_demo = trips_demo.assign(crnt_tur = d[crnt_tur])
    trips_demo = trips_demo.assign(Tipo_familia = e[Tipo_familia])
    
    x = np.array(trips_demo)
    y_pred = model.predict(x)

    return y_pred

def getPredFullText(pred):
    pred_text = {"Coche": "Private Car", "TP": "Public Transit"}
    return pred_text[pred]

def save_data(Per_hog, Turismos, Edad, crnt_tur, Tipo_familia, model_result, r_w, scenario):
    data = pd.read_csv(f"data/data.csv")

    fila = pd.Series({'Per_hog': Per_hog, 'Turismos': Turismos, 'Edad': Edad, 'crnt_tur': crnt_tur,'Tipo_familia': Tipo_familia, 'Scenario': scenario, 'Model_result': model_result, 'R_W': r_w})
   
    data = pd.concat([data, pd.DataFrame([fila])], ignore_index=True)
    data.to_csv(f"data/data.csv", index=False)

with open(file_path, 'rb') as file:
    model = pickle.load(file)

pred = ["1","2"]

def main():
    #data = pd.DataFrame(columns=['Per_hog', 'Turismos', 'Edad', 'crnt_tur', 'Tipo_familia', 'Scenario', 'Model_result', 'R_W'])
    #data.to_csv(f"data/data.csv", index=False)
    
    current_phase = 0

    phases = {0: "Data Input", 1: "Scenario 1", 2: "Scenario 2", 3: "Scenario 3"}

    trips_demo = createDataframe()
    
    st.markdown("<h1 style='color: white; text-align: center;padding-bottom: 0px;'>Predicting Commuting Behaviors</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: white; text-align: center;font-weight: ligshter; padding-bottom: 0px;'>A Mobility Choice Model for Gipuzkoa by MIT City Science</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='color: grey; text-align: center;font-weight: lighter; padding-bottom: 100px;'><i>Iñigo Azcarate, Naroa Coretti, Diego Antonelli</i></h2>", unsafe_allow_html=True)
    

    with st.sidebar:
        st.markdown("<h1 style='color: white'>Tell us about yourself...</h1>", unsafe_allow_html=True)

        # User input for profile data
        #age = st.select_slider("Age", options=["<19", "20-29", "30-44", "45-59", "60-74", "75+"])

        # Create two columns
        #col1, col2, col3, col4 = st.columns(4)

        # Place the select_slider in the first column
        age = st.select_slider("Age", options=["<19", "20-29", "30-44", "45-59", "60-74", "75+"])
        persons_per_family = st.select_slider("Person/family", options=["1-2", "3-5", "6+"])
        number_of_cars = st.select_slider("Number of cars", options=["0", "1", "2", "3+"])
        has_driver_license = st.radio("Drivers licence", ["Yes", "No"])
        family_type = st.radio("Family type", ["Single", "Single parent", "Couple", "Couple with kids"])

        if st.button("SAVE DATA"):
            # Placeholder for prediction logic
            pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        
    # SCENARIOS IMAGE 
    video_file = open('images/Central-SS.mov', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.markdown("<h1 style='color: white; text-align: center; padding-bottom: 100px; padding-top: 100px;'>How would you commute?</h1>", unsafe_allow_html=True)

    st.markdown("<h4 style='color: white; text-align: center; padding-top: 100px;'>Scenario 1</h1>", unsafe_allow_html=True)
    image = Image.open('images/scenario1.png')
    st.image(image)
    if st.button("Reveal Model Prediction 1"):
        st.text("Scenario 1 Result")
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        st.markdown("<h5 style='color: green; text-align: left;'>"+getPredFullText(pred[0])+"</h2>", unsafe_allow_html=True)
    st.text("The model was...")
    cols = st.columns(8)
    right_button = cols[0].button("Right")
    wrong_button = cols[1].button("Wrong")

    if right_button:
        st.write("Thank you for your feedback!")
        st.balloons()
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        save_data(persons_per_family, number_of_cars, age, has_driver_license, family_type, pred[0], "right", "Scenario 1")

    if wrong_button:
        st.write("Sorry, we'll try to improve!")
        st.markdown("😢" * 3)
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        save_data(persons_per_family, number_of_cars, age, has_driver_license, family_type, pred[0], "wrong", "Scenario 1")

    
    st.markdown("<h4 style='color: white; text-align: center; padding-top: 100px;'>Scenario 2</h1>", unsafe_allow_html=True)
    image = Image.open('images/scenario2.png')
    st.image(image)
    if st.button("Reveal Model Prediction 2"):
        st.text("Scenario 2 Result")
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        st.markdown("<h5 style='color: green; text-align: left;'>"+getPredFullText(pred[1])+"</h2>", unsafe_allow_html=True)
    st.text("The model was...")
    cols = st.columns(8)
    right_button = cols[0].button("Right ")
    wrong_button = cols[1].button("Wrong ")

    if right_button:
        st.write("Thank you for your feedback!")
        st.balloons()
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        save_data(persons_per_family, number_of_cars, age, has_driver_license, family_type, pred[1], "right", "Scenario 2")

    if wrong_button:
        st.write("Sorry, we'll try to improve!")
        st.markdown("😢" * 3)
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        save_data(persons_per_family, number_of_cars, age, has_driver_license, family_type, pred[1], "wrong", "Scenario 2")




    st.markdown("<h4 style='color: white; text-align: center; padding-top: 100px;'>Scenario 3</h1>", unsafe_allow_html=True)
    image = Image.open('images/scenario3.png')

    st.image(image)
    if st.button("Reveal Model Prediction 3"):
        st.text("Scenario 3 Result")
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        st.markdown("<h5 style='color: green; text-align: left;'>"+getPredFullText(pred[2])+"</h2>", unsafe_allow_html=True)
    st.text("The model was...")
    cols = st.columns(8)
    right_button = cols[0].button("Right  ")
    wrong_button = cols[1].button("Wrong  ")

    if right_button:
        st.write("Thank you for your feedback!")
        st.balloons()
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        save_data(persons_per_family, number_of_cars, age, has_driver_license, family_type, pred[2], "right", "Scenario 3")

    if wrong_button:
        st.write("Sorry, we'll try to improve!")
        st.markdown("😢" * 3)
        pred = predict_demo(persons_per_family, number_of_cars, age, has_driver_license, family_type, model, trips_demo)
        save_data(persons_per_family, number_of_cars, age, has_driver_license, family_type, pred[2], "wrong", "Scenario 3")   
        
            

    

       

if __name__ == '__main__':
    main()

