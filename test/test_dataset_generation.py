import multiprocessing
import time
from src.python.dataset_generation.dataset_creation import collect_dataset
from src.python.dataset_generation.data_augmentation import random_rotation

if __name__ == "__main__":
    class_names =  ['01_Highfive', '02_Fist', '03_Peace', '04_Fingerguns', '05_ThumbsUp']
    for label in class_names:
        directory = f"./Dataset/{label}"
        p = multiprocessing.Process(target=collect_dataset, name="Collect Dataset", args=(directory,))
        p.start()

        #Run code for 30 seconds before moving to next class
        time.sleep(30)

        #Terminate function
        p.terminate()
        p.join()
    
    status = random_rotation("./Dataset")
    print(status)