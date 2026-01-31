# ✍️ Handwritten Text Recognition (HTR)

## 📌 Project Overview
Handwritten Text Recognition (HTR) is the process of converting handwritten text images into machine-readable digital text.  
This project implements a complete preprocessing and text recognition pipeline designed to handle variations in handwriting styles, noise, skew, and document quality.

The system focuses on robust image preprocessing, accurate line segmentation, and reliable handwritten text extraction.

---

## 🎯 Objectives
- Convert handwritten text images into digital text
- Handle noise, skew, and contrast issues in scanned documents
- Improve text clarity using advanced preprocessing techniques
- Accurately segment handwritten text lines
- Support real-world handwritten document digitization

---

## 🧠 Methodology
The system follows a multi-stage handwritten text recognition pipeline:

1. Input Processing  
2. Image Preprocessing  
3. Noise Removal  
4. Contrast Enhancement  
5. Thresholding  
6. Morphological Operations  
7. Noise Filtering  
8. Skew Detection and Correction  
9. Line Segmentation and Extraction  

---

## ⚙️ Processing Pipeline

### 1. Input Processing
- Converts uploaded image bytes into numerical arrays
- Image decoding and matrix formation
- Libraries: NumPy, OpenCV

### 2. Grayscale Conversion
- Converts RGB images into grayscale
- Reduces complexity while preserving text information

### 3. Multi-Stage Noise Removal
- Median filtering
- Bilateral filtering
- Preserves text edges while removing noise

### 4. Contrast Enhancement
- Histogram normalization
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Enhances faded or low-contrast text

### 5. Thresholding
- Otsu’s global thresholding
- Adaptive local thresholding
- Converts grayscale image to binary format

### 6. Morphological Operations
- Morphological closing
- Uses elliptical kernels to connect broken characters

### 7. Noise Filtering
- Area-based filtering
- Aspect ratio filtering
- Removes non-text components

### 8. Skew Detection and Correction
- Canny edge detection
- Hough line transform
- Affine rotation to align text horizontally

### 9. Line Segmentation and Extraction
- Horizontal projection analysis
- Detects text line boundaries
- Extracts and resizes individual text lines

---

## 📊 Results
- Improved handwritten text clarity
- Accurate line segmentation
- Effective handling of noisy and skewed documents
- Suitable for integration with OCR and deep learning HTR models

---

## 🚀 Applications
- Digitization of historical manuscripts
- Automated exam paper evaluation
- Bank cheque processing
- Medical handwritten note transcription
- Legal document analysis
- Postal address recognition

---

## 🧪 Technologies Used
- Python
- OpenCV
- NumPy
- PIL
- Matplotlib

---

