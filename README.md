# 🖼️ Image Captioning and Translation Pipeline

This project presents a complete pipeline that takes an **image as input**, generates a **descriptive caption in English**, and then translates that caption into **Farsi**.  
It serves as a practical example of combining **state-of-the-art computer vision** and **natural language processing models**.

---

## 🔧 Core Technologies

This pipeline is built upon two powerful deep learning models:

- **Image Captioning (ClipCap)**  
  Uses the **ClipCap model architecture**, which connects the visual understanding of OpenAI's **CLIP** model with the text-generation capabilities of a **GPT-2** language model.  
  It translates the image's content into a meaningful prefix that guides the language model to generate a relevant caption.

- **Translation (SeamlessM4T v2)**  
  For translation, the project leverages Meta AI's **SeamlessM4T v2**, a multilingual and multitask model.  
  It is highly effective for translating text between numerous languages. Here, it is used to convert generated **English captions into Farsi**.

---

## ⚙️ How It Works

The process is orchestrated by the main script and can be broken down into the following steps:

1. **Input** – The user provides an image via a command-line argument (URL or local file path).  
2. **Image Loading** – The script fetches and loads the image into a format suitable for processing.  
3. **Caption Generation** – The `ImageCaptioner` extracts visual features using **CLIP**, passes them through the **ClipCap projection network**, and generates an English caption with **GPT-2**.  
4. **Translation** – The `TranslationModel` uses **SeamlessM4T** to translate the English caption into **Farsi**.  
5. **Output** – Both captions are printed to the console.

---

## 📂 Project Structure

```
.
├── main.py             # Main script to run the entire pipeline
├── captioner.py        # Contains the ClipCap model and ImageCaptioner class
├── translator.py       # Wrapper for the SeamlessM4T translation model
├── utils.py            # Helper functions for downloading models and loading images
├── requirements.txt    # A list of all Python dependencies for the project
└── README.md           # This documentation file
```

---

## ⚡ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create a Python Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the script from your terminal.

**The first time you run it:**
- The captioner weights (coco_weights.pkl, ~235MB) will be downloaded.
- The translation model will also be downloaded automatically by the transformers library.

### Example with a URL
```bash
python main.py "https://i.ytimg.com/vi/vEyP6J61H4s/maxresdefault.jpg"
```

### Example with a Local File
```bash
python main.py "./images/my_photo.jpg"
```

### Options
Prevent the script from opening a window to display the input image:
```bash
python main.py "path/to/your/image.jpg" --no-display
```

---

## 📋 Example Output

```bash
> python main.py "https://i.ytimg.com/vi/vEyP6J61H4s/maxresdefault.jpg"

Loading image from: https://i.ytimg.com/vi/vEyP6J61H4s/maxresdefault.jpg
Model weights already exist at coco_weights.pkl.
Loading translation model...
Translation model loaded.

Generating caption...
  [English Caption]: a cat sitting on a couch with a remote control

Translating caption to Farsi...
  [Farsi Translation]: یه گربه روی مبل با کنترل از راه دور نشسته
```

---

## 📸 Demo Screenshots

Here are 8 examples of the pipeline in action.  
(Each image contains its English caption and Farsi translation inside the image itself.)

<p align="center">
  <img src="./images/img1.jpg" width="22%" />
  <img src="./images/img2.jpg" width="22%" />
  <img src="./images/img3.jpg" width="22%" />
  <img src="./images/img4.jpg" width="22%" />
</p>

<p align="center">
  <img src="./images/img5.jpg" width="22%" />
  <img src="./images/img6.jpg" width="22%" />
  <img src="./images/img7.jpg" width="22%" />
  <img src="./images/img8.jpg" width="22%" />
</p>

---

## 📌 License

This project is released under the MIT License.
