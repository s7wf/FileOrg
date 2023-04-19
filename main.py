import fnmatch
import os
import sys

import exifread
import torch
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QMessageBox,
                             QProgressBar, QTreeView, QPlainTextEdit, QAbstractItemView)
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from custom_text_classification import CustomTextClassificationPipeline

sys.setrecursionlimit(10000)

if os.name == 'nt' and sys.getwindowsversion()[0] >= 6:
    # Set the manifest to request elevated permissions
    try:
        import ctypes
        ctypes.windll.kernel32.SetDllDirectoryW(None)
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
        myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        exe = sys.executable
        if hasattr(sys, '_MEIPASS'):
            exe = os.path.join(sys._MEIPASS, os.path.basename(sys.executable))
        # Change the filename to your application's main executable file
        ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, None, None, 1)
        sys.exit(0)
    except:
        pass


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Set up the UI
        layout = QVBoxLayout()

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english")

        model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
        device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        self.classifier = CustomTextClassificationPipeline(
            model=model,
            tokenizer=self.tokenizer,
            device=device,
            framework="pt"
        )
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

    def generate_tags(self, file_path):
        tags = []

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
        stripped_content = [line.strip() for line in content]

        print(f"Content: {stripped_content}")  # Add this line for debugging purposes

        results = self.classifier(stripped_content)

        # Extract the labels with a threshold confidence score
        threshold = 0.5
        for result in results:
            if result['score'] > threshold:
                tags.append(result['label'])

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
