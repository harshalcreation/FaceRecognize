import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import numpy as np
from playsound import playsound
import csv
import threading
from datetime import datetime

# Known face encodings and names
known_face_encodings = []
known_face_names = []
recognized_faces = {}

# Load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    print("Loading known faces...")
    person_images = ["person1.jpg", "person2.jpg"]
    person_names = ["John Doe", "Jane Doe"]

    for img_path, name in zip(person_images, person_names):
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)
        print(f"Loaded {name} from {img_path}")

# Face recognition function
def start_surveillance():
    print("Starting surveillance...")
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        messagebox.showerror("Error", "Unable to access the camera.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video. Exiting...")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Unknown"
            accuracy = 0

            if face_distances[best_match_index] <= 0.6:  # Threshold for recognition
                name = known_face_names[best_match_index]
                accuracy = (1 - face_distances[best_match_index]) * 100

                # Record to CSV if face not recorded in the past hour
                now = datetime.now()
                if name not in recognized_faces or (now - recognized_faces[name]).seconds > 3600:
                    recognized_faces[name] = now
                    record_to_csv(name, accuracy, now)
                    playsound("alert.mp3")  # Play alert sound

            # Draw a rectangle and name with accuracy
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({accuracy:.2f}%)",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Record face recognition data to CSV
def record_to_csv(name, accuracy, timestamp):
    with open("recognized_faces.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, f"{accuracy:.2f}%", timestamp.strftime("%Y-%m-%d %H:%M:%S")])
        print(f"Recorded {name} ({accuracy:.2f}%) at {timestamp}")

# Function to start surveillance in a separate thread
def start_surveillance_thread():
    threading.Thread(target=start_surveillance, daemon=True).start()

# Create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Face Recognition Surveillance")
    root.geometry("600x400")
    root.configure(bg="#f4f4f9")

    # Title
    title_label = tk.Label(root, text="Face Recognition Surveillance", font=("Arial", 20, "bold"), bg="#f4f4f9", fg="#333")
    title_label.pack(pady=20)

    # Start Button
    start_button = tk.Button(root, text="Start Surveillance", font=("Arial", 14), bg="#007bff", fg="white", command=start_surveillance_thread)
    start_button.pack(pady=10)

    # About Us Section
    about_frame = tk.Frame(root, bg="#f4f4f9")
    about_frame.pack(pady=20)
    about_label = tk.Label(about_frame, text="About Us", font=("Arial", 16, "bold"), bg="#f4f4f9", fg="#555")
    about_label.pack(anchor="w", padx=20)

    about_text = tk.Label(
        about_frame,
        text="This project is created by Team Blinkers for the Madhya Pradesh Police.\nIt provides real-time face recognition surveillance with CSV logging and accuracy analysis.",
        font=("Arial", 12),
        bg="#f4f4f9",
        fg="#555",
        justify="left",
    )
    about_text.pack(anchor="w", padx=20)

    # Footer
    footer = tk.Label(root, text="© 2024 Team Blinkers | Contact: blinkers@domain.com", font=("Arial", 10), bg="#f4f4f9", fg="#999")
    footer.pack(side="bottom", pady=10)

    # Start GUI main loop
    root.mainloop()

if __name__ == "__main__":
    load_known_faces()
    create_gui()
