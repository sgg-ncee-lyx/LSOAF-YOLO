from ultralytics.models import YOLO
import os

if __name__ == '__main__':

    # Load a model
    model = YOLO("ultralytics/cfg/models/LSOAF-YOLO.yaml").load("yolo11n.pt")  

    # Train the model
    results = model.train(
        data="visdrone.yaml",
        epochs=100,
        imgsz=640,  #  影响质量
        batch=4, #影响质量
        device=0, #GPU 'cpu'
        amp=False,
        patience=50,
        project='runs',
        save_json=True,
    )


