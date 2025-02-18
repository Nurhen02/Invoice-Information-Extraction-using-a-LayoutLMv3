import sqlite3  # For interacting with SQLite databases
import json  # For reading JSON files (annotations)
import os  # For file and directory operations
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting datasets into training and testing sets
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer  # For LayoutLMv3 model and training utilities
from datasets import Dataset, Features, Sequence, Value, Array2D  # For creating and managing datasets in Hugging Face format
from PIL import Image  # For image processing
import torch  # PyTorch, the deep learning framework
from seqeval.metrics import classification_report, accuracy_score  # For evaluating sequence labeling tasks
import re  # For regular expressions (used in date validation)
import warnings  # For suppressing warnings
import pytesseract  # For Optical Character Recognition (OCR) to extract text from images
from pytesseract import Output  # For structured OCR output

warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore UserWarnings

class Config:
    IMAGE_DIR = "invoices/images"  # Directory containing invoice images
    ANNOTATION_DIR = "invoices/Annotations/layoutlm_HF_format"  # Directory containing annotations in LayoutLM format
    DB_PATH = "extracted_invoices.db"  # Path to the SQLite database for storing extracted invoice data
    
    MODEL_NAME = "microsoft/layoutlmv3-base"  # Pre-trained LayoutLMv3 model to use
    LABEL_NAMES = [  # List of labels for Named Entity Recognition (NER)
        "O", "B-DATE", "I-DATE",  # "O" = Outside, "B" = Beginning, "I" = Inside
        "B-BUYER", "I-BUYER",  # Buyer entity
        "B-TOTAL", "I-TOTAL",  # Total amount entity
        "B-TAX", "I-TAX",  # Tax amount entity
        "OTHER"  # Default label for unrecognized entities
    ]
    NUM_LABELS = len(LABEL_NAMES)  # Number of unique labels
    ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}  # Mapping from label IDs to label names
    LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}  # Mapping from label names to label IDs
    
    # Training Parameters
    BATCH_SIZE = 4  # Number of samples per batch during training
    LEARNING_RATE = 1e-5  # Learning rate for the optimizer
    NUM_EPOCHS = 3  # Number of training epochs
    WEIGHT_DECAY = 0.1  # Regularization parameter to prevent overfitting
    MAX_SEQ_LENGTH = 512  # Maximum sequence length for the model
    TEST_SIZE = 0.3  # Fraction of data to use for testing
    DROPOUT_RATE = 0.4  # Dropout rate for regularization

class InvoiceDataset:
    def __init__(self, annotation_dir, image_dir):
        self.annotation_dir = annotation_dir  # Directory containing annotation files
        self.image_dir = image_dir  # Directory containing invoice images
        self.processor = LayoutLMv3Processor.from_pretrained(Config.MODEL_NAME, apply_ocr=False)  # Initialize LayoutLMv3 processor

    def load_data(self, limit=1500):
        samples = []  # List to store processed samples
        annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.json')][:limit]  # Get list of annotation files (limit to 1500)
        
        for ann_file in annotation_files:  # Iterate through each annotation file
            with open(os.path.join(self.annotation_dir, ann_file)) as f:  # Open the annotation file
                data = json.load(f)  # Load JSON data
                
            # Skip duplicates by checking if the image path already exists in samples
            if any(s['image_path'] == data['path'] for s in samples):
                continue  # Skip duplicates
                
            image_path = os.path.join(self.image_dir, data['path'])  # Get full path to the image
            image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB format
            
            # Validate and convert NER tags
            valid_ner_tags = []  # List to store valid NER tags
            for tag in data['ner_tags']:  # Iterate through each NER tag
                if tag >= Config.NUM_LABELS or tag < 0:  # Check if tag is invalid
                    valid_ner_tags.append(Config.LABEL2ID["OTHER"])  # Replace invalid tags with "OTHER"
                else:
                    valid_ner_tags.append(tag)  # Keep valid tags

            # Process the image, words, bounding boxes, and labels using the LayoutLMv3 processor
            encoding = self.processor(
                image,  # Input image
                data['words'],  # List of words extracted from the image
                boxes=data['bboxes'],  # Bounding boxes for each word
                word_labels=valid_ner_tags,  # Validated NER tags
                truncation=True,  # Truncate sequences longer than MAX_SEQ_LENGTH
                padding="max_length",  # Pad sequences to MAX_SEQ_LENGTH
                max_length=Config.MAX_SEQ_LENGTH,  # Maximum sequence length
                return_offsets_mapping=True,  # Return offset mappings for alignment
                return_tensors="pt"  # Return PyTorch tensors
            )
            
            # Append processed sample to the list
            samples.append({
                'id': data['path'],  # Unique identifier (image path)
                'input_ids': encoding['input_ids'].squeeze(),  # Token IDs
                'attention_mask': encoding['attention_mask'].squeeze(),  # Attention mask
                'bbox': encoding['bbox'].squeeze(),  # Bounding boxes
                'labels': encoding['labels'].squeeze(),  # NER labels
                'image_path': image_path  # Path to the image
            })
        return samples  # Return the list of processed samples

def split_dataset(dataset):
    # Stratify by presence of key entities (e.g., DATE, BUYER, TOTAL, TAX)
    labels = [1 if any(tag in [1,3,5,7] for tag in sample['labels']) else 0 
              for sample in dataset]
    
    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(
        dataset, 
        test_size=Config.TEST_SIZE,  # Fraction of data to use for testing
        stratify=labels,  # Ensure balanced distribution of key entities
        random_state=42  # Seed for reproducibility
    )
    return train_data, test_data  # Return training and testing datasets

class InvoiceModelTrainer:
    def __init__(self):
        self.processor = LayoutLMv3Processor.from_pretrained(Config.MODEL_NAME)  # Initialize LayoutLMv3 processor
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=Config.NUM_LABELS,  # Number of unique labels
            id2label=Config.ID2LABEL,  # Mapping from label IDs to label names
            label2id=Config.LABEL2ID,  # Mapping from label names to label IDs
            hidden_dropout_prob=Config.DROPOUT_RATE,  # Dropout rate for hidden layers
            attention_probs_dropout_prob=Config.DROPOUT_RATE  # Dropout rate for attention layers
        )

    def compute_metrics(self, p):
        predictions, labels = p  # Unpack predictions and true labels
        predictions = np.argmax(predictions, axis=2)  # Get predicted class IDs (shape: [batch_size, seq_length])

        # Filter out special tokens (e.g., padding) and invalid labels
        true_predictions = []  # Stores valid predictions per sample
        true_labels = []  # Stores valid true labels per sample
        
        for prediction, label in zip(predictions, labels):  # Iterate over each sample
            valid_preds = []  # Valid predictions for current sample
            valid_lbls = []  # Valid true labels for current sample
            
            for p, l in zip(prediction, label):  # Iterate over each token
                # Skip padding tokens (label = -100) and "O" labels
                if l != -100 and Config.ID2LABEL.get(p, "O") != "O":
                    valid_preds.append(Config.ID2LABEL[p])  # Convert ID to label name
                    valid_lbls.append(Config.ID2LABEL[l])  # Convert ID to label name
            
            true_predictions.append(valid_preds)  # Add filtered predictions
            true_labels.append(valid_lbls)  # Add filtered labels

        # Calculate metrics using seqeval
        return {
            "accuracy": accuracy_score(true_labels, true_predictions),  # Overall accuracy
            **classification_report(true_labels, true_predictions, output_dict=True)  # Detailed metrics (precision, recall, F1)
        }

    def train(self, train_data, test_data):
        # Define dataset schema for Hugging Face Datasets
        features = Features({
            'input_ids': Sequence(Value(dtype='int64')),  # Token IDs
            'attention_mask': Sequence(Value(dtype='int64')),  # Attention masks
            'bbox': Array2D(dtype="int64", shape=(Config.MAX_SEQ_LENGTH, 4)),  # Bounding boxes
            'labels': Sequence(Value(dtype='int64'))  # NER labels
        })
        
        # Convert training data to Hugging Face Dataset format
        train_dataset = Dataset.from_dict({
            'input_ids': [sample['input_ids'].tolist() for sample in train_data],  # Convert tensors to lists
            'attention_mask': [sample['attention_mask'].tolist() for sample in train_data],
            'bbox': [sample['bbox'].tolist() for sample in train_data],
            'labels': [sample['labels'].tolist() for sample in train_data]
        }, features=features)

        # Convert test data to Hugging Face Dataset format (same structure as train)
        test_dataset = Dataset.from_dict({
            'input_ids': [sample['input_ids'].tolist() for sample in test_data],
            'attention_mask': [sample['attention_mask'].tolist() for sample in test_data],
            'bbox': [sample['bbox'].tolist() for sample in test_data],
            'labels': [sample['labels'].tolist() for sample in test_data]
        }, features=features)

        # Configure training parameters
        training_args = TrainingArguments(
            output_dir="./results",  # Directory to save checkpoints and logs
            num_train_epochs=Config.NUM_EPOCHS,  # Number of training epochs
            per_device_train_batch_size=Config.BATCH_SIZE,  # Batch size per GPU
            per_device_eval_batch_size=Config.BATCH_SIZE,  # Batch size for evaluation
            learning_rate=Config.LEARNING_RATE,  # Learning rate
            weight_decay=Config.WEIGHT_DECAY,  # Weight decay for regularization
            evaluation_strategy="epoch",  # Evaluate after each epoch
            save_strategy="epoch",  # Save model after each epoch
            logging_dir='./logs',  # Directory for training logs
            load_best_model_at_end=True,  # Load the best model at the end of training
            save_total_limit=2,  # Maximum number of checkpoints to keep
            metric_for_best_model="eval_loss",  # Metric to determine the best model
            greater_is_better=False,  # Lower eval_loss is better
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,  # Model to train
            args=training_args,  # Training configuration
            train_dataset=train_dataset,  # Training data
            eval_dataset=test_dataset,  # Evaluation data
            compute_metrics=self.compute_metrics,  # Custom metrics function
        )

        trainer.train()  # Start training
        return trainer  # Return trained model and metrics

class InvoiceProcessor:
    def __init__(self, model_path):
        self.processor = LayoutLMv3Processor.from_pretrained(Config.MODEL_NAME, apply_ocr=False)  # Initialize processor
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)  # Load trained model
        self.conn = sqlite3.connect(Config.DB_PATH)  # Connect to SQLite database
        self._init_db()  # Initialize database table

    def _init_db(self):
        """Create the invoices table if it doesn't exist"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  # Auto-incrementing ID
                date TEXT,  # Extracted date
                buyer TEXT,  # Extracted buyer name
                total REAL,  # Extracted total amount
                tax REAL,  # Extracted tax amount
                image_path TEXT  # Path to the original image
            )''')
        self.conn.commit()  # Commit schema changes

    def process_invoice(self, image_path):
        image = Image.open(image_path).convert("RGB")  # Load and convert image
        
        # Step 1: Perform OCR to extract words and bounding boxes
        ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
        words = []  # List to store extracted words
        boxes = []  # List to store bounding boxes
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text:  # Skip empty strings
                x = ocr_data['left'][i]  # Left coordinate
                y = ocr_data['top'][i]  # Top coordinate
                w = ocr_data['width'][i]  # Width
                h = ocr_data['height'][i]  # Height
                boxes.append([x, y, x + w, y + h])  # Store as [x0, y0, x1, y1]
                words.append(text)  # Store cleaned text

        # Step 2: Normalize bounding boxes to [0, 1000] scale
        image_width, image_height = image.size  # Get image dimensions
        normalized_boxes = []  # Stores normalized boxes
        
        for box in boxes:
            normalized_box = [
                int(1000 * (box[0] / image_width)),  # x0
                int(1000 * (box[1] / image_height)),  # y0
                int(1000 * (box[2] / image_width)),  # x1
                int(1000 * (box[3] / image_height)),  # y1
            ]
            normalized_boxes.append(normalized_box)  # Add normalized box

        # Step 3: Process with LayoutLMv3
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",  # Return PyTorch tensors
            truncation=True,  # Truncate to MAX_SEQ_LENGTH
            padding="max_length",  # Pad to MAX_SEQ_LENGTH
            max_length=Config.MAX_SEQ_LENGTH,
            return_offsets_mapping=True,  # Needed for token-to-word alignment
        )

        # Remove offset_mapping (not used by the model)
        model_inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
        }

        # Step 4: Get predictions
        with torch.no_grad():  # Disable gradient calculation
            outputs = self.model(**model_inputs)  # Forward pass
        predictions = outputs.logits.argmax(-1).squeeze().tolist()  # Get predicted label IDs

        # Step 5: Align predictions with original words
        offset_mapping = encoding['offset_mapping'].squeeze().tolist()  # Token-to-word mapping
        word_predictions = []  # Stores predictions at word level
        current_word_id = -1  # Tracks current word index

        for i, (offset, pred) in enumerate(zip(offset_mapping, predictions)):
            if offset == (0, 0):  # Skip special tokens (e.g., [CLS], [SEP])
                continue
            
            # Detect new word when token starts at a new position
            if i == 0 or offset[0] != offset_mapping[i-1][1]:
                current_word_id += 1
                if current_word_id >= len(words):  # Handle edge case
                    continue
                word_predictions.append(Config.ID2LABEL.get(pred, "O"))  # Map ID to label

        # Step 6: Extract entities from word predictions
        entities = {"date": "", "buyer": "", "total": "", "tax": ""}  # Initialize entity store
        current_entity = None  # Tracks active entity (e.g., "DATE", "BUYER")
        
        for word, label in zip(words, word_predictions):
            if label.startswith("B-"):
                current_entity = label.split("-")[1].lower()  # Start new entity (e.g., "B-DATE" → "date")
                entities[current_entity] = word  # Initialize entity value
            elif label.startswith("I-") and current_entity:  # Continue existing entity
                entities[current_entity] += f" {word}"  # Append word to entity
            else:
                current_entity

    def _validate_numeric(self, value):
        """Convert numeric fields to float, default to 0.0 on failure"""
        if isinstance(value, (int, float)):  # Already numeric
            return float(value)
        
        try:
            cleaned = str(value).replace('$', '').replace(',', '').replace('€', '').strip()  # Remove currency symbols
            return float(cleaned)  # Convert to float
        except (ValueError, TypeError):  # Handle invalid formats
            return 0.0  # Default value

    def _validate_date(self, value):
        """Validate date format (DD-MM-YYYY)"""
        if isinstance(value, str):
            if re.match(r"\d{2}-\d{2}-\d{4}", value):  # Regex check for date format
                return value
        return ""  # Return empty string for invalid dates

    def save_to_db(self, entities, image_path):
        """Save extracted entities to SQLite database"""
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO invoices (date, buyer, total, tax, image_path)
                          VALUES (?, ?, ?, ?, ?)''',  # Parameterized query
                       (entities.get('date', ''),
                        entities.get('buyer', ''),
                        entities.get('total', 0.0),
                        entities.get('tax', 0.0),
                        image_path))
        self.conn.commit()  # Commit transaction

    def print_db_contents(self):
        """Print all rows from the invoices table"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM invoices")  # Fetch all records
        rows = cursor.fetchall()
        
        print("\n***** Extracted Invoices Database Contents *****")
        print("ID | Date       | Buyer         | Total  | Tax   | Image Path")
        print("-" * 60)
        for row in rows:
            print(f"{row[0]:<3} | {row[1]:<10} | {row[2]:<13} | {row[3]:<6.2f} | {row[4]:<5.2f} | {row[5]}")  # Formatted output                                            

# Check if the script is being executed directly (not imported as a module)
if __name__ == "__main__":
    # Load invoice dataset with duplicate check
    dataset = InvoiceDataset(Config.ANNOTATION_DIR, Config.IMAGE_DIR).load_data()
    
    # Verify dataset diversity by counting unique invoice IDs
    unique_dates = len(set([sample['id'] for sample in dataset]))
    print(f"Loaded {len(dataset)} invoices ({unique_dates} unique templates)")

    # Split dataset into training and testing sets
    train_data, test_data = split_dataset(dataset)

    # Verify there is no data leakage between training and testing sets
    train_ids = {sample['id'] for sample in train_data}  # Extract IDs from training data
    test_ids = {sample['id'] for sample in test_data}  # Extract IDs from test data
    
    # Ensure no overlap (assertion will raise an error if data leakage is found)
    assert len(train_ids & test_ids) == 0, "Data leakage detected!"  # No common samples allowed

    # Train the LayoutLMv3 model using training data
    trainer = InvoiceModelTrainer().train(train_data, test_data)

    # Define path to save the best trained model
    best_model_path = "./results/best_model"
    
    # Save the best trained model for later use in inference
    trainer.save_model(best_model_path)

    # Evaluate model performance on the test dataset
    test_results = trainer.predict(test_data)  # Get evaluation metrics
    
    # Display test set evaluation results
    print("\n***** Test Set Evaluation *****")
    print(f"Accuracy: {test_results.metrics['test_accuracy'] * 100:.1f}%")
    print(f"Precision: {test_results.metrics['test_weighted avg']['precision'] * 100:.1f}%")
    print(f"Recall: {test_results.metrics['test_weighted avg']['recall'] * 100:.1f}%")
    print(f"F1-score: {test_results.metrics['test_weighted avg']['f1-score'] * 100:.1f}%")

    # Create an invoice processor using the trained model
    processor = InvoiceProcessor(best_model_path)

    # Process and save the first 50 invoices from the test dataset into the database
    for sample in test_data[:50]:  # Only process the first 50 test samples
        entities = processor.process_invoice(sample['image_path'])  # Extract entities from invoice
        processor.save_to_db(entities, sample['image_path'])  # Store extracted data in the database

    # Print all stored invoice records from the database
    processor.print_db_contents()

    # Display confirmation message with database file location
    print(f"\nData saved to {Config.DB_PATH}")