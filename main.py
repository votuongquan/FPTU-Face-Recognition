import os
import cv2 as cv2
from tkinter import Tk
import face_recognition as fr
from tkinter.filedialog import askopenfilename
                
class FaceRecognition:
    def __init__(self, filepath):
        self.root = Tk()
        self.root.withdraw()
        self.filepath = filepath
        self.load_image = askopenfilename()
        self.target_image = fr.load_image_file(self.load_image)
        self.target_encodings = fr.face_encodings(self.target_image)

    def encode_faces(self, folder):
        list_people_encoding = []
        
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            known_image = fr.load_image_file(file_path)
            if len(fr.face_encodings(known_image)) == 0:
                continue
            known_encoding = fr.face_encodings(known_image)[0]

            list_people_encoding.append((known_encoding, filename.split('.')[0]))
            
        return list_people_encoding

    def find_target_face(self):
        face_locations = fr.face_locations(self.target_image)

        for person in self.encode_faces(self.filepath):
            if len(self.target_encodings) == 0:
                print("No faces found in the image.")
                return
            encoded_face = person[0]
            filename = person[1]
            
            is_target_face = fr.compare_faces(encoded_face, self.target_encodings, tolerance=0.50)
            
            face_distances = fr.face_distance(encoded_face, self.target_encodings)
            accuracy = 1 - face_distances[0]
            print(f'{is_target_face} {filename} {accuracy}')
            
            if face_locations:
                face_number = 0
                for location in face_locations:
                    if is_target_face[face_number]:
                        label = f'{filename}'
                        self.create_frame(location, label, accuracy)
                    face_number += 1
            else:
                print("No faces found in the image.")
                return

    def create_frame(self, location, filename, accuracy):
        top, right, bottom, left = location
        cv2.rectangle(self.target_image, (left, top), (right, bottom), (255, 0, 0), 3)
        cv2.putText(self.target_image, filename, (left, top-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        accuracy_percentage = "{:.2f}%".format(accuracy * 100)
        cv2.putText(self.target_image, accuracy_percentage, (left, bottom+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    def render_image(self):
        rgb_img = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Face Recognition', rgb_img)
        cv2.waitKey(0)

if __name__ == "__main__":
    source_image_path = f'E:\Major\SP24\CPV301\Face_Recog\Image'
    face_recognition = FaceRecognition(source_image_path)
    face_recognition.find_target_face()
    face_recognition.render_image()