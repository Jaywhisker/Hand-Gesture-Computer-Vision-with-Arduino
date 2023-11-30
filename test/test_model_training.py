from src.python.model_training.training_model import *

if __name__ == "__main__":
    datasetfilepath = "./Dataset"
    modelpath = "./Models/best_model.h5"

    class_names, train_ds, val_ds = preparing_dataset(datasetfilepath)
    train_model(class_names, train_ds, val_ds,modelpath)
    visualise_dataset(train_ds)
