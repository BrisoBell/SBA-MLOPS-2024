
import json
from PIL import Image
import io
import os
import numpy as np
import streamlit as st
from streamlit import session_state
from tensorflow.keras.models import load_model #type: ignore
from keras.preprocessing import image as img_preprocessing #type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input #type: ignore
import base64
from tensorflow import keras


session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0


@st.cache_resource
def load_model():
    model = keras.models.load_model("models//EfficientNetB7_model.keras")
    return model

def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def predict(image_path):
    model = load_model()
    img = img_preprocessing.load_img(image_path, target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    classes = ['Anemia', 'Non_Anemia']
    return classes[np.argmax(predictions)]


def generate_medical_report(predicted_label):
    # Define class labels and corresponding medical information
    medical_info = {
        "Non_Anemia": {
            "report": (
                "Good news! The patient does not show signs of anemia. "
                "Maintaining a healthy lifestyle is key to keeping their hemoglobin levels within a normal range. "
                "Regular monitoring and a balanced diet can help sustain this healthy state."
            ),
            "preventative_measures": [
                "Continue with a balanced diet rich in iron, vitamin B12, and folic acid: These nutrients are essential for the production of hemoglobin and healthy red blood cells.",
                "Stay hydrated and maintain a healthy weight: Proper hydration and a healthy weight support overall bodily functions and reduce the risk of developing anemia.",
                "Regular physical activity: Exercise improves circulation and overall health, helping to maintain healthy hemoglobin levels."
            ],
            "precautionary_measures": [
                "Keep up with routine health check-ups: Regular visits to a healthcare provider ensure that any potential issues are detected early.",
                "Monitor for any symptoms of anemia such as fatigue, weakness, or pale skin: Early detection of symptoms can help prevent the condition from worsening."
            ],
        },
        "Anemia": {
            "report": (
                "The patient has been diagnosed with anemia. "
                "It's important to address this condition promptly to avoid complications such as severe fatigue and weakness. "
                "Following medical advice and making lifestyle changes can significantly improve the patient's health."
            ),
            "preventative_measures": [
                "Increase dietary intake of iron-rich foods such as red meat, beans, and leafy green vegetables: Iron is crucial for the production of hemoglobin, which carries oxygen in the blood.",
                "Consider iron supplements as recommended by a healthcare provider: Supplements can help quickly replenish iron stores in the body.",
                "Ensure adequate intake of vitamin C to enhance iron absorption: Vitamin C improves the absorption of iron from plant-based sources, enhancing its effectiveness."
            ],
            "precautionary_measures": [
                "Schedule regular follow-up appointments to monitor hemoglobin levels: Regular monitoring helps track the effectiveness of treatment and dietary changes.",
                "Consult with a healthcare provider for personalized treatment options: A healthcare provider can offer tailored advice and treatment plans based on the specific type and cause of anemia."
            ],
        },
    }

    # Retrieve medical information based on predicted label
    medical_report = medical_info[predicted_label]["report"]
    preventative_measures = medical_info[predicted_label]["preventative_measures"]
    precautionary_measures = medical_info[predicted_label]["precautionary_measures"]

    # Generate conversational medical report with each section in a paragraphic fashion
    report = (
        "Medical Report:\n\n"
        + medical_report
        + "\n\nPreventative Measures:\n\n- "
        + "\n- ".join(preventative_measures)
        + "\n\nPrecautionary Measures:\n\n- "
        + "\n- ".join(precautionary_measures)
    )
    precautions = precautionary_measures

    return report, precautions



def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")



def save_image(image_file, json_file_path="data.json"):
    try:
        if image_file is None:
            st.warning("No file uploaded.")
            return

        if not session_state["logged_in"] or not session_state["user_info"]:
            st.warning("Please log in before uploading images.")
            return

        # Load user data from JSON file
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        # Find the user's information
        for user_info in data["users"]:
            if user_info["email"] == session_state["user_info"]["email"]:
                image = Image.open(image_file)

                if image.mode == "RGBA":
                    image = image.convert("RGB")

                # Convert image bytes to Base64-encoded string
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

                # Update the user's information with the Base64-encoded image string
                user_info["finger_nail"] = image_base64

                # Save the updated data to JSON
                with open(json_file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)

                session_state["user_info"]["finger_nail"] = image_base64
                return

        st.error("User not found.")
    except Exception as e:
        st.error(f"Error saving finger_nail image to JSON: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
        email = email.lower()
        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "report": None,
            "precautions": None,
            "finger_nail":None

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None



def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    username = username.lower()
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        # Title and user information
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information")
        st.markdown(f"""
        **Name:** {user_info['name']}  
        **Sex:** {user_info['sex']}  
        **Age:** {user_info['age']}  
        """)

        # Open the JSON file and check for the 'finger_nail' key
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == user_info["email"]:
                    if "finger_nail" in user and user["finger_nail"] is not None:
                        image_data = base64.b64decode(user["finger_nail"])
                        st.image(Image.open(io.BytesIO(image_data)), caption="Uploaded Fingernail Image", use_column_width=True)
                    else:
                        st.warning("No fingernail image uploaded. Please upload an image for a more accurate report.")

                    # Display medical report and precautions
                    if "report" in user_info and "precautions" in user_info and user_info["report"] is not None and user_info["precautions"] is not None:
                        st.subheader("Medical Report")
                        st.info(user_info["report"])

                        st.subheader("Precautions")
                        for precaution in user_info["precautions"]:
                            st.write(f"- {precaution}")
                    else:
                        st.warning("Reminder: Please upload fingernail images and generate a report.")
        

        
        with st.expander("More Information About Anemia"):
            st.write("""
            **Anemia** is a condition where you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues. 
            Anemia can make you feel tired and weak. There are many forms of anemia, each with its own cause. Anemia can be temporary or long term, and it can range from mild to severe.
            
            **Common Causes of Anemia**:
            - Iron deficiency
            - Vitamin B12 deficiency
            - Chronic diseases
            - Genetic conditions

            **Symptoms of Anemia**:
            - Fatigue
            - Weakness
            - Pale or yellowish skin
            - Irregular heartbeats
            - Shortness of breath
            - Dizziness or lightheadedness
            - Chest pain
            - Cold hands and feet
            - Headaches
            """)
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
        
        
def fetch_precautions(user_info):
    return (
        user_info["precautions"]
        if user_info["precautions"] is not None
        else "Please upload fingernail images and generate a report."
    )


def main(json_file_path="data.json"):
    st.sidebar.title("Anemia Detection System")
    page = st.sidebar.selectbox(
        "Go to",
        ("Signup/Login", "Dashboard", "Upload Fingernail Image", "View Reports"),
        key="key",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Upload Fingernail Image":
        if session_state.get("logged_in"):
            st.title("Upload Fingernail Image")
            uploaded_image = st.file_uploader(
                "Choose a Fingernail image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
            )
            if st.button("Upload") and uploaded_image is not None:
                st.image(uploaded_image, use_column_width=True)
                st.success("Fingernail image uploaded successfully!")
                save_image(uploaded_image, json_file_path)
                with st.spinner("Detecting the presence of Anemia..."):
                    condition = predict(uploaded_image)
                with st.spinner("Generating medical report..."):
                    report, precautions = generate_medical_report(condition)

                # Read the JSON file, update user info, and write back to the file
                with open(json_file_path, "r+") as json_file:
                    data = json.load(json_file)
                    user_index = next((i for i, user in enumerate(data["users"]) if user["email"] == session_state["user_info"]["email"]), None)
                    if user_index is not None:
                        user_info = data["users"][user_index]
                        user_info["report"] = report
                        user_info["precautions"] = precautions
                        session_state["user_info"] = user_info
                        json_file.seek(0)
                        json.dump(data, json_file, indent=4)
                        json_file.truncate()
                    else:
                        st.error("User not found.")
                st.write(report)
        else:
            st.warning("Please login/signup to upload a Fingernail image.")

    elif page == "View Reports":
        if session_state.get("logged_in"):
            st.title("View Reports")
            user_info = get_user_info(session_state["user_info"]["email"], json_file_path)
            if user_info is not None:
                if user_info["report"] is not None:
                    st.subheader("Fingernail Report:")
                    st.write(user_info["report"])
                else:
                    st.warning("No reports available.")
            else:
                st.warning("User information not found.")
        else:
            st.warning("Please login/signup to view reports.")



if __name__ == "__main__":
    initialize_database()
    main()
