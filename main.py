import fnmatch
import os
import sys

import exifread
import torch
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QMessageBox,
                             QProgressBar, QTreeView, QPlainTextEdit, QAbstractItemView)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torchvision
import torchvision.transforms as transforms
from PIL import Image

sys.setrecursionlimit(10000)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Set up the UI
        layout = QVBoxLayout()

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", max_length=128)
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        self.scan_button = QPushButton("Scan Files")
        self.scan_button.clicked.connect(self.on_scan_files_button_clicked)
        layout.addWidget(self.scan_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.tree_view = QTreeView()
        self.model = QStandardItemModel()
        self.tree_view.setModel(self.model)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.tree_view)
        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)
        layout.addWidget(self.output_box)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.statusBar = self.statusBar()
        # For text classification
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", max_length=128)
        self.text_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

        # For image object detection
        self.image_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.image_model.eval()

    def get_image_metadata(self, image_path):
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            return tags

    def traverse_directory(self, directory):
        files = []
        for root, _, file_names in os.walk(directory):
            for file_name in file_names:
                files.append(os.path.join(root, file_name))
        return files

    def detect_objects(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)
        with torch.no_grad():
            predictions = self.image_model([image])
        objects = []
        for pred in predictions[0]['labels']:
            label = self.image_model.classes[pred]
            objects.append(label)
        return objects

    def classify_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        inputs = self.text_tokenizer(content, return_tensors="pt", truncation=True, padding=True)
        outputs = self.text_model(**inputs)
        label_index = torch.argmax(outputs.logits, dim=1).item()
        label = self.text_tokenizer.convert_ids_to_tokens([label_index])[0]
        return [label]

    def generate_tags(self, file_path):
        tags = []
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            objects = self.detect_objects(file_path)
            tags.extend(objects)
        elif file_path.endswith(('.txt', '.md', '.pdf', '.docx')):
            label = self.classify_text(file_path)
            tags.extend(label)
        return tags

    def scan_and_tag_files(self):
        global file_path
        personal_folders = [
            #os.path.expanduser('~/Documents'),
            #os.path.expanduser('~/Pictures'),
            #os.path.expanduser('~/Downloads'),
            os.path.expanduser('~/Desktop')
            # Add more personal folders to scan
        ]
        filters = ['*.txt', '*.md', '*.pdf', '*.docx', '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.mp4', '*.avi',
                   '*.mkv']
        files = []
        for folder in personal_folders:
            for file_filter in filters:
                for root, _, file_names in os.walk(folder):
                    for file_name in file_names:
                        if fnmatch.fnmatch(file_name, file_filter):
                            file_path = os.path.join(root, file_name)
                            files.append(file_path)

        total_files = len(files)
        print(f"Total files to scan: {total_files}")

        for index, file in enumerate(files):
            tags = self.generate_tags(file)

            # Add the file and tags to the model
            file_item = QStandardItem(file)
            tags_item = QStandardItem(', '.join(tags))
            self.model.appendRow([file_item, tags_item])

            progress = int((index + 1) / total_files * 100)
            self.progress_bar.setValue(progress)
            QApplication.processEvents()

            # Print progress in the output box
            self.output_box.appendPlainText(
                f"Scanned file {index + 1}/{total_files}: {file} - Tags: {', '.join(tags)}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

    def on_scan_files_button_clicked(self):
        print("Scan files button clicked.")
        reply = QMessageBox.question(self, 'Scan Files',
                                     'This process might take some time. Do you want to continue?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            print("User clicked Yes.")
            self.output_box.clear()
            self.statusBar.showMessage("Scanning files. Please wait...")
            self.scan_and_tag_files()
            self.progress_bar.setValue(0)
            self.statusBar.clearMessage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
