from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data='/home/heman/research-project/datasets/TR-prediction/data.yaml',  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640  # training image size # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("overheat/simulated_images/20w/heatmap_1050-1060.png")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model