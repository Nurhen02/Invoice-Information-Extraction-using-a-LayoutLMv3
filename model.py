import sqlite3
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, Features, Sequence, Value, Array2D
from PIL import Image
import torch
from seqeval.metrics import classification_report, accuracy_score
import re
import warnings
import pytesseract
from pytesseract import Output
from dateutil import parser

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Config:
    IMAGE_DIR = "invoices/images"
    ANNOTATION_DIR = "invoices/Annotations/layoutlm_HF_format"
    DB_PATH = "extracted_invoices.db"
    
    MODEL_NAME = "microsoft/layoutlmv3-base"
    
    LABEL_NAMES = [
        "O",           # 0
        "B-TOTAL",     # 1
        "I-TOTAL",     # 2
        "B-DATE",      # 3
        "I-DATE",      # 4
        "B-BUYER",     # 5
        "I-BUYER",     # 6
        "B-TAX",       # 7
        "I-TAX",       # 8
        "B-INVOICE",   # 9
        "I-INVOICE",   # 10
        "B-STRUCTURE", # 11
        "I-STRUCTURE", # 12
        "OTHER"        # 13
    ]
    NUM_LABELS = len(LABEL_NAMES)
    ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}
    LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}
    
    # Enhanced training parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 10  # Increased epochs
    WEIGHT_DECAY = 0.1
    MAX_SEQ_LENGTH = 512
    TEST_SIZE = 0.3
    DROPOUT_RATE = 0.4

class InvoiceDataset:
    def __init__(self, annotation_dir, image_dir):
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.processor = LayoutLMv3Processor.from_pretrained(Config.MODEL_NAME, apply_ocr=False)

    def load_data(self, limit=50):
        samples = []
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.json')][:limit]
        
        for ann_file in annotation_files:
            with open(os.path.join(self.annotation_dir, ann_file)) as f:
                data = json.load(f)
                
            if any(s['image_path'] == data['path'] for s in samples):
                continue
                
            image_path = os.path.join(self.image_dir, data['path'])
            image = Image.open(image_path).convert("RGB")
            
            # Validate label indices
            valid_ner_tags = []
            for tag in data['ner_tags']:
                if tag >= Config.NUM_LABELS or tag < 0:
                    valid_ner_tags.append(Config.LABEL2ID["OTHER"])
                else:
                    valid_ner_tags.append(tag)

            encoding = self.processor(
                image,
                data['words'],
                boxes=data['bboxes'],
                word_labels=valid_ner_tags,
                truncation=True,
                padding="max_length",
                max_length=Config.MAX_SEQ_LENGTH,
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            
            samples.append({
                'id': data['path'],
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'bbox': encoding['bbox'].squeeze(),
                'labels': encoding['labels'].squeeze(),
                'image_path': image_path
            })
        return samples

def split_dataset(dataset):
    # Create stratification labels
    labels = [
        1 if any(
            tag in [
                Config.LABEL2ID["B-TOTAL"], 
                Config.LABEL2ID["B-DATE"],
                Config.LABEL2ID["B-BUYER"], 
                Config.LABEL2ID["B-TAX"]
            ] 
            for tag in sample['labels']
        ) 
        else 0 
        for sample in dataset
    ]
    
    # First split: 80% train+val, 20% test
    train_val_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    
    # Get labels for train_val subset
    train_val_indices = [i for i, sample in enumerate(dataset) if sample in train_val_data]
    train_val_labels = [labels[i] for i in train_val_indices]
    
    # Second split: 60% train, 20% val
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=0.25,
        stratify=train_val_labels,
        random_state=42
    )
    
    return train_data, val_data, test_data  # Now returns 3 datasets!

class InvoiceModelTrainer:
    def __init__(self):
        self.processor = LayoutLMv3Processor.from_pretrained(Config.MODEL_NAME)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=Config.NUM_LABELS,
            id2label=Config.ID2LABEL,
            label2id=Config.LABEL2ID,
            hidden_dropout_prob=Config.DROPOUT_RATE,
            attention_probs_dropout_prob=Config.DROPOUT_RATE
        )

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            valid_preds = []
            valid_lbls = []
            
            for p, l in zip(prediction, label):
                if l != -100 and Config.ID2LABEL.get(p, "O") != "O":
                    valid_preds.append(Config.ID2LABEL[p])
                    valid_lbls.append(Config.ID2LABEL[l])
            
            true_predictions.append(valid_preds)
            true_labels.append(valid_lbls)

        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            **classification_report(true_labels, true_predictions, output_dict=True)
        }

    def train(self, train_data, val_data):  # Now takes val_data instead of test_data
        features = Features({
            'input_ids': Sequence(Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(Config.MAX_SEQ_LENGTH, 4)),
            'labels': Sequence(Value(dtype='int64'))
        })
        
        # Convert all datasets
        train_dataset = Dataset.from_dict({
            'input_ids': [s['input_ids'].tolist() for s in train_data],
            'attention_mask': [s['attention_mask'].tolist() for s in train_data],
            'bbox': [s['bbox'].tolist() for s in train_data],
            'labels': [s['labels'].tolist() for s in train_data]
        }, features=features)

        val_dataset = Dataset.from_dict({
            'input_ids': [s['input_ids'].tolist() for s in val_data],
            'attention_mask': [s['attention_mask'].tolist() for s in val_data],
            'bbox': [s['bbox'].tolist() for s in val_data],
            'labels': [s['labels'].tolist() for s in val_data]
        }, features=features)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=Config.NUM_EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            load_best_model_at_end=True,
            save_total_limit=2,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        return trainer # Return trained model and metrics

class InvoiceProcessor:
    def __init__(self, model_path):
        self.processor = LayoutLMv3Processor.from_pretrained(Config.MODEL_NAME, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.conn = sqlite3.connect(Config.DB_PATH)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            buyer TEXT,
            total REAL,
            tax REAL,
            image_path TEXT
        )''')
        self.conn.commit()

    def process_invoice(self, image_path):
        image = Image.open(image_path).convert("RGB")
        
        # Improved OCR with PSM 6
        ocr_data = pytesseract.image_to_data(
            image,
            config='--psm 6 --oem 3',
            output_type=Output.DICT
        )
        
        words = []
        boxes = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text:
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                boxes.append([x, y, x + w, y + h])
                words.append(text)

        # Validate box normalization
        image_width, image_height = image.size
        normalized_boxes = []
        for box in boxes:
            if image_width == 0 or image_height == 0:
                continue  # Prevent division by zero
            normalized_box = [
                max(0, min(1000, int(1000 * (box[0] / image_width)))),
                max(0, min(1000, int(1000 * (box[1] / image_height)))),
                max(0, min(1000, int(1000 * (box[2] / image_width)))),
                max(0, min(1000, int(1000 * (box[3] / image_height)))),
            ]
            normalized_boxes.append(normalized_box)

        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_SEQ_LENGTH,
            return_offsets_mapping=True,
        )

        model_inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()

        offset_mapping = encoding['offset_mapping'].squeeze().tolist()
        word_predictions = []
        current_word_id = -1

        for i, (offset, pred) in enumerate(zip(offset_mapping, predictions)):
            if offset == (0, 0):
                continue
            
            if i == 0 or offset[0] != offset_mapping[i-1][1]:
                current_word_id += 1
                if current_word_id >= len(words):
                    continue
                word_predictions.append(Config.ID2LABEL.get(pred, "O"))

        # Fixed entity extraction logic
        entities = {"date": "", "buyer": "", "total": "", "tax": ""}
        current_entity = None
        current_value = []
        
        for word, label in zip(words, word_predictions):
            if label.startswith("B-"):
                if current_entity:
                    entities[current_entity] = " ".join(current_value)
                    current_value = []
                current_entity = label.split("-")[1].lower()
                current_value.append(word)
            elif label.startswith("I-") and current_entity:
                current_value.append(word)
            else:
                if current_entity:
                    entities[current_entity] = " ".join(current_value)
                    current_entity = None
                    current_value = []
        
        if current_entity:
            entities[current_entity] = " ".join(current_value)

        # Enhanced validation
        entities["total"] = self._validate_numeric(entities["total"])
        entities["tax"] = self._validate_numeric(entities["tax"])
        entities["date"] = self._validate_date(entities["date"])
        
        return entities

    def _validate_numeric(self, value):
        try:
            cleaned = re.sub(r'[^\d.]', '', str(value))
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0

    def _validate_date(self, value):
        try:
            return parser.parse(value, fuzzy=True).strftime("%d-%b-%Y")
        except:
            return ""

    def save_to_db(self, entities, image_path):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO invoices (date, buyer, total, tax, image_path)
                          VALUES (?, ?, ?, ?, ?)''',
                       (entities.get('date', ''),
                        entities.get('buyer', ''),
                        entities.get('total', 0.0),
                        entities.get('tax', 0.0),
                        image_path))
        self.conn.commit()

    def print_db_contents(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM invoices")
        rows = cursor.fetchall()
        
        print("\n***** Extracted Invoices Database Contents *****")
        print("ID | Date       | Buyer           | Total   | Tax   | Image Path")
        print("-" * 70)
        for row in rows:
            print(f"{row[0]:<3} | {row[1]:<10} | {row[2]:<15} | {row[3]:<7.2f} | {row[4]:<5.2f} | {row[5]}")

# Check if the script is being executed directly (not imported as a module)
if __name__ == "__main__":
    # Load invoice dataset with duplicate check
    dataset = InvoiceDataset(Config.ANNOTATION_DIR, Config.IMAGE_DIR).load_data()
    
    # Verify dataset diversity by counting unique invoice IDs
    unique_templates = len(set([sample['id'] for sample in dataset]))
    print(f"Loaded {len(dataset)} invoices ({unique_templates} unique templates)")

    # Split dataset into three parts (train/val/test)
    train_data, val_data, test_data = split_dataset(dataset)

    # Verify no data leakage between any sets
    train_ids = {sample['id'] for sample in train_data}
    val_ids = {sample['id'] for sample in val_data}
    test_ids = {sample['id'] for sample in test_data}
    
    assert train_ids.isdisjoint(val_ids), "Data leakage: Train/Val overlap detected!"
    assert train_ids.isdisjoint(test_ids), "Data leakage: Train/Test overlap detected!"
    assert val_ids.isdisjoint(test_ids), "Data leakage: Val/Test overlap detected!"

    # Initialize and train the model
    model_trainer = InvoiceModelTrainer()
    huggingface_trainer = model_trainer.train(train_data, val_data)  # Get actual Trainer instance

    # Save the best model
    best_model_path = "./results/best_model"
    huggingface_trainer.save_model(best_model_path)

    # Prepare test dataset for final evaluation
    test_features = Features({
        'input_ids': Sequence(Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(Config.MAX_SEQ_LENGTH, 4)),
        'labels': Sequence(Value(dtype='int64'))
    })
    
    test_dataset = Dataset.from_dict({
        'input_ids': [sample['input_ids'].tolist() for sample in test_data],
        'attention_mask': [sample['attention_mask'].tolist() for sample in test_data],
        'bbox': [sample['bbox'].tolist() for sample in test_data],
        'labels': [sample['labels'].tolist() for sample in test_data]
    }, features=test_features)

    # Final evaluation on test set
    test_results = huggingface_trainer.evaluate(test_dataset)
    print("\n***** Final Test Set Evaluation *****")
    print(f"Accuracy: {test_results['eval_accuracy'] * 100:.1f}%")
    print(f"Precision: {test_results['eval_weighted avg']['precision'] * 100:.1f}%")
    print(f"Recall: {test_results['eval_weighted avg']['recall'] * 100:.1f}%")
    print(f"F1-score: {test_results['eval_weighted avg']['f1-score'] * 100:.1f}%")

    # Process and store test samples
    processor = InvoiceProcessor(best_model_path)
    for sample in test_data[:50]:
        entities = processor.process_invoice(sample['image_path'])
        processor.save_to_db(entities, sample['image_path'])

    # Display database contents
    processor.print_db_contents()
    print(f"\nData saved to {Config.DB_PATH}")
