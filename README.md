# Invoice Information Extraction using LayoutLMv3

## 1. Introduction
This project is a Deep Learning-based solution for extracting key information from invoices, such as the date, buyer name, total amount, and tax amount. It utilizes **LayoutLMv3**, a state-of-the-art transformer model designed to process scanned documents while considering both textual and layout information.

## 2. Why LayoutLMv3?
**LayoutLMv3** is specifically designed for document understanding tasks. Unlike traditional OCR-based approaches, LayoutLMv3:
- **Understands both text and layout** (spatial positioning of words)
- **Improves information extraction from structured documents** such as invoices and receipts
- **Leverages transformer-based deep learning** to improve accuracy in extracting key fields like dates, names, and amounts

We chose LayoutLMv3 because it is optimized for invoices, where both textual and positional data are crucial.

## 3. Dataset and Annotation Format
The dataset consists of invoice images and corresponding annotation files stored in the **layoutlm_HF_format**. Each annotation file contains:
- `path`: The name of the invoice image
- `ner_tags`: Labels indicating the type of each word (e.g., DATE, TOTAL, TAX)
- `words`: Extracted words from the invoice
- `bboxes`: Bounding box coordinates of words in the image

## 4.Key Library References
Transformers: Provides pre-trained models and training utilities (Hugging Face)

Datasets: Dataset handling and preprocessing (Hugging Face)

SQLite3: Lightweight disk-based database (Python standard library)

PyTorch: Deep learning framework

Seqeval: Evaluation metrics for sequence labeling

Scikit-learn: Machine learning utilities (data splitting)

PIL: Image processing

NumPy: Numerical computations

## 5. How the Code Works

### Step 1: Configuration
The `Config` class defines essential settings, including:
- Paths for images and annotation files
- Model settings (e.g., `microsoft/layoutlmv3-base`)
- Training parameters like batch size, learning rate, and epochs

### Step 2: Data Processing
The `InvoiceDataset` class:
- Loads invoice images and annotations
- Uses **LayoutLMv3Processor** to tokenize and encode text while preserving layout information
- Prepares the dataset for training

### Step 3: Dataset Splitting
The function `split_dataset()` ensures balanced training and testing sets while avoiding data leakage.

### Step 4: Model Training
The `InvoiceModelTrainer` class:
- Loads the **pre-trained LayoutLMv3 model**
- Fine-tunes it using the annotated dataset
- Evaluates performance using accuracy and classification reports
- Saves the best-performing model

### Step 5: Invoice Information Extraction & Storage
The `InvoiceProcessor` class:
- Uses the trained model to extract key details (date, buyer, total, tax)
- Cleans and validates extracted values
- Saves extracted data to an **SQLite database**

### Step 6: Structured Entity Recognition with OCR
This project integrates Optical Character Recognition (OCR) using Tesseract to extract text from invoice images. The extracted text and their bounding box coordinates are then fed into LayoutLMv3 for structured entity recognition.
OCR Processing Steps:

1. Extract text from the invoice image using pytesseract.image_to_data().
2. Obtain bounding box coordinates for each recognized word.
3. Normalize bounding boxes to a 0-1000 scale for compatibility with LayoutLMv3.
4. Tokenize and process the text using LayoutLMv3Processor.
5. Use the trained model to classify words into entities like DATE, BUYER, TOTAL, TAX.
6. Store structured invoice data in an SQLite database.


## 7. Expected Output
Once trained, the model can process invoices and extract:

```
Extracted Invoices Database Contents 

ID | Date       | Buyer        | Total  | Tax  | Image Path
-----------------------------------------------------------
1  | 27-Aug-1993| Jose Graham  | 853.49 | 0.00 | Template10_Instance28.jpg
2  | 20-Mar-2008| Denise Perez | 734.33 | 28.18| Template1_Instance0.jpg
```

## 8. How to Run the Project
1. Install dependencies using:
   ```
   pip install -r requirements.txt
   ```
2. Place invoice images inside `invoices/images`.
3. Place annotation JSON files inside `invoices/Annotations/layoutlm_HF_format`.
4. Run the main script:
   ```
   python model.py
   ```
5. The trained model will be saved, and extracted invoice data will be stored in `extracted_invoices.db`.

## 9. Conclusion
This project automates invoice processing using deep learning. It eliminates manual data entry and enables seamless integration into business workflows. Future improvements could include training on more datasets and integrating an external OCR system for better text detection.

## 10. REMARK
The Dataset is not available in this Project.

