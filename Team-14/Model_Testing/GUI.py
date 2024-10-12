import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk

# Define the harmful object classes
HARMFUL_OBJECT_CLASSES = {
    0: "Knife",
    1: "Rifle",
    2: "Gun",
    3: "Cigarette",
    4: "Alcohol"
}

class HarmfulObjectDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transforms=None):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.transforms = transforms
        
        self.img_names = [img for img in sorted(os.listdir(self.imgs_path)) if img.endswith(('.jpg', '.png', '.jpeg'))]
        self.label_names = [label for label in sorted(os.listdir(self.labels_path)) if label.endswith('.txt')]
        
        self.img_names = [img for img in self.img_names if self._get_label_file(img)]

    def _get_label_file(self, img_name):
        base_name = img_name.rsplit('.', 1)[0]
        label_file = f"{base_name}.txt"
        return label_file if label_file in self.label_names else None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.imgs_path, img_name)
        label_file = self._get_label_file(img_name)
        label_path = os.path.join(self.labels_path, label_file)
        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_rgb /= 255.0
        
        if self.transforms:
            img_rgb = self.transforms(img_rgb)
        
        with open(label_path, 'r') as f:
            line = f.readline().strip().split()
            label = int(line[0])
            x_center, y_center, width, height = map(float, line[1:])
        
        img_height, img_width = img_rgb.shape[1], img_rgb.shape[2]
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        x_max = (x_center + width / 2) * img_width
        y_max = (y_center + height / 2) * img_height

        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid bounding box: ({x_min}, {y_min}, {x_max}, {y_max})")
        
        boxes = torch.as_tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        labels = torch.as_tensor([label], dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        return img_rgb, target

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def make_predictions(model, image, device):
    model.eval()
    with torch.no_grad():
        prediction = model(image)
    return prediction[0]

def plot_prediction(image, prediction, threshold=0.5):
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    img_np = np.array(image)

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_np)

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 edgecolor='r', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'{HARMFUL_OBJECT_CLASSES[label]}: {score:.2f}', color='white', fontsize=12, backgroundcolor='black')

    plt.axis('off')
    plt.show()

class ObjectDetectionApp:
    def __init__(self, root, model_path):
        self.root = root
        self.model_path = model_path
        self.device = torch.device("cpu")  # Use CPU
        self.num_classes = len(HARMFUL_OBJECT_CLASSES)  # Number of classes in your model
        self.model = get_model(self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.label = Label(root, text="Upload an Image to Detect Harmful Objects")
        self.label.pack(pady=20)

        self.upload_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = T.ToTensor()(img_rgb).unsqueeze(0).to(self.device)

        # Make predictions
        prediction = make_predictions(self.model, img_tensor, self.device)

        # Plot results and show in the canvas
        self.plot_result(img_rgb, prediction)

    def plot_result(self, image, prediction):
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score > 0.5:  # Set a threshold for detection
                x_min, y_min, x_max, y_max = box
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     edgecolor='r', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                ax.text(x_min, y_min, f'{HARMFUL_OBJECT_CLASSES[label]}: {score:.2f}', color='white', fontsize=12, backgroundcolor='black')

        plt.axis('off')

        # Save the result as an image to display on the canvas
        plt.savefig("temp_result.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Load and display the result image
        self.display_image("temp_result.png")

    def display_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((800, 600), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

if __name__ == "__main__":
    root = Tk()
    root.title("Harmful Object Detection")
    model_path = 'harmful_object_detector.pth'  # Path to your trained model
    app = ObjectDetectionApp(root, model_path)
    root.mainloop()
