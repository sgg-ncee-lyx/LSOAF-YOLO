from ultralytics.models import YOLO

if __name__ == '__main__':

    model = YOLO("runs/train/weights/best.pt")
    results = model.val(
        data='visdrone.yaml',
        project='VisDrone_Val',
        save_json=True,
    )




