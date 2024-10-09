# Product Category Prediction using CLIP

This repository contains a Python script that processes a CSV file containing image URLs, downloads the images, and uses OpenAI's CLIP model to predict product categories. The predictions are then saved to a new CSV file. The script is optimized for parallel downloading and efficient batch processing.

## Installation

### Local Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/product-category-clip.git
   cd product-category-clip
# Image Category Prediction with CLIP in Google Colab

## Overview

This notebook performs the following tasks:

1. **Downloads Images**: Fetches images from URLs listed in a CSV file.
2. **Processes Images**: Resizes images to a standard size for model compatibility.
3. **Predicts Categories**: Uses the CLIP model to predict categories for each image based on a predefined list of unique categories.
4. **Saves Results**: Outputs the predictions to a new CSV file.

## Prerequisites

- A Google account to access Google Colab.
- Two CSV files:
  - **`image_url.csv`**: Contains image URLs under the column `PRODUCT_MAIN_IMAGE_URL`.
  - **`product_category.csv`**: Contains product categories under the column `Product_category`.




# Project Title

1. Install Required Libraries

 
!pip install torch torchvision transformers pandas requests Pillow tqdm ```bash

- **Python 3.7 or higher**
- **Google Colab Account** (if running on Colab)
- **CSV Files:**
-**Purpose:** Installs essential libraries needed for the script
-**torch and torchvision:** PyTorch and its image processing utilities.

-**transformers:** For using pre-trained models like CLIP.

-**pandas:** For data manipulation and analysis.

-**requests:** For making HTTP requests to download images.

-**Pillow:** For image processing.

## 2. Check GPU Availability
```bash

from IPython.display import HTML, clear_output
from subprocess import getoutput
s = getoutput('nvidia-smi')
if 'K80' in s:gpu = 'K80'
elif 'T4' in s:gpu = 'T4'
elif 'P100' in s:gpu = 'P100'
else:
    gpu='DONT PROCEED'
display(HTML(f"<h1>{gpu}</h1>"))
```
-**Purpose:** Checks for GPU availability and its type (K80, T4, P100) using nvidia-smi and displays the result.

-**Usage:** Ensures that the code runs on a suitable environment for GPU processing.


## Requirements

This project requires the following Python libraries:

- `torch`
- `torchvision`
- `transformers`
- `pandas`
- `requests`
- `Pillow`
- `tqdm`
## 3. Import Required Libraries
```bash
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import requests
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
torch.cuda.empty_cache()
```

-**`Purpose:`** Imports necessary libraries for image processing, model inference, data handling, and parallel execution.

-**`torch.cuda.empty_cache():`** Frees up unused memory from the GPU.

## 4. Define Constants and Load Data
```bash
CSV_FILE_PATH = '/content/image_url.csv'
BATCH_SIZE = 100
MAX_WORKERS = 20
CACHE_DIR = 'image_cache'

data = pd.read_csv('/content/product_category.csv')
df = pd.DataFrame(data)
unique_categories = df['Product_category'].unique()
unique_categories_list = unique_categories.tolist()
```

-**`CSV_FILE_PATH:`** Path to the CSV file containing image URLs.

-**`BATCH_SIZE:`** Number of images to process in each batch

-**`MAX_WORKERS:`** Maximum number of threads to use for downloading images.

-**`CACHE_DIR:`** Directory to cache downloaded images.

-**`Loads the product categories from another CSV file and creates a list of unique categories.

##5. Load Image URLs from CSV
```bash
try:
    data = pd.read_csv(CSV_FILE_PATH)
    df = pd.DataFrame(data)

    required_columns = ['PRODUCT_MAIN_IMAGE_URL']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The CSV file must contain a column named '{col}'.")

    print(f"Successfully loaded {len(df)} image URLs from the CSV file.")
except FileNotFoundError:
    print(f"CSV file not found at path: {CSV_FILE_PATH}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"CSV file at {CSV_FILE_PATH} is empty.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    exit(1)
```
-**`Purpose:`** Loads image URLs from the specified CSV file and checks for the required column PRODUCT_MAIN_IMAGE_URL.

-**`Error Handling:`** Catches various exceptions (file not found, empty data) and exits if there's an issue.
## 6. Define Function to Load and Cache Images
```bash
def load_image_from_url(url, size=(224, 224), retries=3, backoff=5, cache_dir='image_cache'):
    ...

```

-`Purpose:` Downloads images from URLs, resizes them, and caches them to avoid re-downloading.

-`Parameters:`

  -`url:` The image URL.
  
  -`size:` Desired size to resize the image.
  
  -`retries:` Number of retry attempts for failed downloads.
  
  -`backoff:` Initial wait time for retries.
  
-`cache_dir:` Directory for cached images.

`The function checks for cached images and uses exponential backoff for retries if downloads fail.`
##7. Define Function to Download Images in Parallel
```bash

def download_images(urls, max_workers=20):
    """
    Downloads images in parallel using ThreadPoolExecutor.
    """
    images = [None] * len(urls)

    def fetch_image(idx, url):
        images[idx] = load_image_from_url(url, size=(224, 224), cache_dir=CACHE_DIR)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_image, idx, url) for idx, url in enumerate(urls)]
        for future in as_completed(futures):
            pass  # Errors are already handled in load_image_from_url

    return images

```

-**Purpose: Uses **`ThreadPoolExecutor`** to download images in parallel for improved performance.**

-**Process: Submits download tasks to the executor and handles the results.**

## 8. Load CLIP Model and Processor
```bash
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading CLIP model or processor: {e}")
    exit(1)

```
-**`Purpose:`** Loads the CLIP model and processor, determining whether to use a GPU or CPU based on availability.

## 9. Define Function for Batch Predictions

```bash

def predict_categories_batch(images, categories, model, processor, device):
    if not images:
        return []

    # Process inputs
    try:
        inputs = processor(text=categories, images=images, return_tensors="pt", padding=True).to(device)
    except Exception as e:
        print(f"Error during processing inputs: {e}")
        return ["Error"] * len(images)

    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: (batch_size, num_categories)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return ["Error"] * len(images)

    # Calculate probabilities
    try:
        probs = logits_per_image.softmax(dim=1).cpu().numpy()  # Shape: (batch_size, num_categories)
    except Exception as e:
        print(f"Error during probability calculation: {e}")
        return ["Error"] * len(images)

    # Get the category with the highest probability for each image
    try:
        predicted_indices = probs.argmax(axis=1)  # Shape: (batch_size,)
        predicted_categories = [categories[idx] for idx in predicted_indices]
    except Exception as e:
        print(f"Error during category assignment: {e}")
        predicted_categories = ["Error"] * len(images)

    return predicted_categories


```
-**`Purpose:`** Takes a batch of images and predicts their categories using the loaded CLIP model.

### Process:

-**`Processes images and categories, performs inference, and calculates probabilities.`** 

-**`Returns the predicted categories.`** 

 ## 10. Process Images in Batches
 ```bash
predicted_categories = []

num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in tqdm(range(num_batches), desc="Processing Batches"):
    start_idx = batch_num * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(df))

    batch_urls = df['PRODUCT_MAIN_IMAGE_URL'].iloc[start_idx:end_idx].tolist()
    
    images = download_images(batch_urls, max_workers=MAX_WORKERS)

    valid_images = []
    valid_indices = []

    for idx, img in enumerate(images):
        if img is not None:
            valid_images.append(img)
            valid_indices.append(idx)
        else:
            predicted_categories.append("Error")

    if valid_images:
        batch_predictions = predict_categories_batch(valid_images, unique_categories_list, model, processor, device)
        for pred in batch_predictions:
            predicted_categories.append(pred)

    while len(predicted_categories) < end_idx:
        predicted_categories.append("Error")


```


**`Purpose:`** Iterates over the DataFrame in batches, downloading images and predicting their categories.

 ### Process:

-**`Downloads images for the current batch.`**

-**`Collects valid images and their corresponding indices.`**

-**`Calls the prediction function for valid images and appends results.`**

## 11. Save Predictions to CSV
```bash
df['predicted_category'] = predicted_categories[:len(df)]

output_csv_path = "product_categories_predictions.csv"
try:
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
except Exception as e:
    print(f"Error saving predictions to CSV: {e}")


```

-**`Purpose:`** Assigns the predicted categories to the DataFrame and saves it as a new CSV file.

-**`Error Handling:`** Catches any exceptions that occur while saving.

 ## 12. Optional: Display the DataFrame

```bash
print(df.head(150))

```




