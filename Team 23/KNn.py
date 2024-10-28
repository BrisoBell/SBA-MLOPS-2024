import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image, ImageTk  # Import Pillow for image handling

# Load the dataset from a CSV file
df = pd.read_csv('csv_result-Autism_Data.csv')

# Clean up the dataset by selecting relevant columns
df_cleaned = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
                 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'Class/ASD']]

# Label encode the target variable 'Class/ASD' (YES/NO)
label_encoder = LabelEncoder()
df_cleaned['Class/ASD'] = label_encoder.fit_transform(df_cleaned['Class/ASD'])

# Split the data into features (X) and target (y)
X = df_cleaned.drop(columns=['Class/ASD'])
y = df_cleaned['Class/ASD']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Calculate predictions and performance metrics
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# AQ-10 Questions (Autism Quotient Test)
AQ10_QUESTIONS = [
    "I often notice small sounds when others do not.",
    "I usually concentrate more on the whole picture, rather than the small details.",
    "I find it easy to do more than one thing at once.",
    "If there is an interruption, I can switch back to what I was doing very quickly.",
    "I find it easy to 'read between the lines' when someone is talking to me.",
    "I know how to tell if someone listening to me is getting bored.",
    "When I’m reading a story, I find it difficult to work out the characters’ intentions.",
    "I like to collect information about categories of things (e.g., types of cars, birds, trains, plants).",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "I find it difficult to work out people’s intentions."
]

# Create a Tkinter window for the GUI
class AutismTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Autism Spectrum Disorder (ASD) Prediction using KNN")
        self.root.geometry("600x400")

        # Set a background image
        self.background_image = Image.open("backgroundd.jpg")
        self.background_image = self.background_image.resize((600, 400))
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(root, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)

        # Create entry for name and age
        self.name_label = tk.Label(root, text="Enter your name:", font=("Arial", 12), bg='lightblue')
        self.name_label.pack(pady=10)

        self.name_entry = tk.Entry(root)
        self.name_entry.pack(pady=5)

        self.age_label = tk.Label(root, text="Enter your age:", font=("Arial", 12), bg='lightblue')
        self.age_label.pack(pady=10)

        self.age_entry = tk.Entry(root)
        self.age_entry.pack(pady=5)

        # Button to start the test
        self.start_button = tk.Button(root, text="Start Test", command=self.start_test, padx=20, pady=10, bg="green", fg="white", font=("Arial", 10, "bold"))
        self.start_button.pack(pady=20)

        # Placeholder for questionnaire
        self.question_index = 0
        self.responses = []

    def start_test(self):
        name = self.name_entry.get()
        age = self.age_entry.get()

        if not name or not age:
            messagebox.showerror("Input Error", "Please enter both name and age.")
            return

        try:
            age = int(age)
        except ValueError:
            messagebox.showerror("Input Error", "Age must be a number.")
            return

        # Remove the name/age inputs and display the first question
        self.name_label.pack_forget()
        self.name_entry.pack_forget()
        self.age_label.pack_forget()
        self.age_entry.pack_forget()
        self.start_button.pack_forget()

        self.question_label = tk.Label(self.root, text=AQ10_QUESTIONS[self.question_index], wraplength=500, font=("Arial", 12), bg='lightblue')
        self.question_label.pack(pady=20)

        # Create "Yes" and "No" buttons
        self.yes_button = tk.Button(self.root, text="Yes", command=lambda: self.collect_response(1), padx=20, pady=10, bg="blue", fg="white", font=("Arial", 10, "bold"))
        self.yes_button.pack(side=tk.LEFT, padx=20, pady=20)

        self.no_button = tk.Button(self.root, text="No", command=lambda: self.collect_response(0), padx=20, pady=10, bg="red", fg="white", font=("Arial", 10, "bold"))
        self.no_button.pack(side=tk.RIGHT, padx=20, pady=20)

    def collect_response(self, response):
        self.responses.append(response)
        self.question_index += 1
        if self.question_index < len(AQ10_QUESTIONS):
            self.question_label.config(text=AQ10_QUESTIONS[self.question_index])
        else:
            self.make_prediction()

    def make_prediction(self):
        input_data = [self.responses]
        prediction = knn.predict(input_data)
        prediction_label = label_encoder.inverse_transform(prediction)

        if prediction_label[0] == 'YES':
            result = "You are autistic."
        else:
            result = "You are not autistic."

        # Show performance metrics along with the prediction
        result_message = (
            f"Prediction: {result}\n"
            f"Model Accuracy: {accuracy * 100:.2f}%\n"
            f"Precision: {precision:.2f}\n"
            f"Recall: {recall:.2f}\n"
            f"F1 Score: {f1:.2f}\n"
            f"Confusion Matrix:\n{conf_matrix}"
        )
        messagebox.showinfo("Prediction Result", result_message)
        self.root.quit()

# Initialize the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AutismTestApp(root)
    root.mainloop()
