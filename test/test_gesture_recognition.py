from src.python.firebase_microservice.recognise_gesture import recognise_hand_gesture

if __name__ == "__main__":
    modelpath = "./Models/best_model.h5"
    recognise_hand_gesture(False, None, None, modelpath)
