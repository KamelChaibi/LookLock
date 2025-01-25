import cv2
import sqlite3
import numpy as np
import face_recognition

def create_database():
    """Initialize the SQLite database and create necessary tables."""
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_user(name, face_encoding):
    """Insert a new user with face encoding into the database."""
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO Users (name, face_encoding) VALUES (?, ?)', (name, face_encoding))
    conn.commit()
    conn.close()

def fetch_users():
    """Retrieve all users and their face encodings from the database."""
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, face_encoding FROM Users')
    rows = cursor.fetchall()
    conn.close()
    return [(name, np.frombuffer(face_encoding, dtype=np.float64)) for name, face_encoding in rows]

def delete_user(name):
    """Delete a user from the database by name."""
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM Users WHERE name = ?', (name,))
    if cursor.rowcount > 0:
        print(f"User '{name}' deleted successfully.")
    else:
        print(f"User '{name}' not found.")
    conn.commit()
    conn.close()

def register_face():
    """Capture a face and register it in the database."""
    name = input("Enter your name: ")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to minimize delay
    print("Position your face in front of the camera...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings_list = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings_list:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_list):
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                insert_user(name, face_encoding.tobytes())
                print(f"User '{name}' registered successfully!")
                cap.release()
                cv2.destroyAllWindows()
                return
        else:
            print("No face detected. Please adjust your position.")

        cv2.imshow('Register Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_face():
    """Capture a face and recognize it using the database."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to minimize delay
    print("Position your face in front of the camera for recognition...")

    users = fetch_users()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings_list = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_list):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            recognized = False
            for name, db_encoding in users:
                matches = face_recognition.compare_faces([db_encoding], face_encoding)
                distance = face_recognition.face_distance([db_encoding], face_encoding)

                if matches[0]:
                    cv2.putText(frame, f"Recognized: {name}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    print(f"Recognized: {name} with distance: {distance[0]:.2f}")
                    recognized = True
                    break

            if not recognized:
                cv2.putText(frame, "Face not recognized", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print("Face not recognized.")

        cv2.imshow('Recognize Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        create_database()

        while True:
            print("1. Register Face")
            print("2. Recognize Face")
            print("3. Delete User")
            print("4. Exit")
            choice = input("Enter your choice: ")

            if choice == '1':
                register_face()
            elif choice == '2':
                recognize_face()
            elif choice == '3':
                name_to_delete = input("Enter the name of the user to delete: ")
                delete_user(name_to_delete)
            elif choice == '4':
                print("Exiting program.")
                break
            else:
                print("Invalid choice. Please try again.")
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
