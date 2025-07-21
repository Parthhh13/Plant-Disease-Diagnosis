# Plant Disease Diagnosis

## Overview

This repository contains a complete, modular, and production-ready pipeline for classifying plant leaf diseases using a combination of classical computer vision, machine learning, deep learning (custom CNN), and transfer learning approaches. The workflow is organized for scalable research, experimentation, and real-world deployment using Streamlit.

## Table of Contents

1. [Introduction](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#introduction)
2. [Features](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#features)
3. [Dataset](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#dataset)
4. [Project Structure](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#project-structure)
5. [Installation](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#installation)
6. [Usage](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#usage)
   * [1. Jupyter/Colab Workflow](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#1-jupytercolab-workflow)
   * [2. Streamlit App](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#2-streamlit-app)
7. [Model Details](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#model-details)
8. [Results and Observations](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#results-and-observations)
9. [Acknowledgments](https://www.perplexity.ai/search/project-title-plant-disease-de-jCJoNf.CSia1c2GmhsayjQ#acknowledgments)

## Introduction

Automated detection of plant diseases is vital for agriculture and food security. This project demonstrates classical and deep learning solutions evaluated on the large PlantVillage dataset, and culminates in a user-friendly real-time prediction web app.

## Features

* **Data Preprocessing** : Consistent feature extraction and pipeline for both classical ML and deep learning.
* **Classical ML (Random Forest)** : Color histograms, texture, and KMeans-based handcrafted features.
* **Custom Deep Learning (CNN)** : End-to-end classifier trained directly on images.
* **(Experimental) Transfer Learning** : EfficientNetB4-based pipeline (not included in deployment due to low validation performance in this case).
* **Streamlit Web App** : Modern UI for image upload and disease prediction using the best model.
* **Modular Code** : Each phase can be run, re-trained, or extended independently.

## Dataset

* **Name:** PlantVillage Dataset
* **Source:** [https://data.mendeley.com/datasets/tywbtsjrjv/1](https://data.mendeley.com/datasets/tywbtsjrjv/1)
* **Format:** Images organized in folders, each corresponding to a particular disease or healthy class.
* **Size:** 54,000+ images across 39 categories.

***Note: The actual dataset is not included in this repository. Please download it manually from the above link and extract it according to project instructions.***

## Project Structure

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-textMainDark selection:text-super selection:bg-super/10 bg-offset my-md relative flex flex-col rounded font-mono text-sm font-thin"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl sticky top-0 flex h-0 items-start justify-end"></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-text-200 bg-background-300 py-xs px-sm inline-block rounded-br rounded-tl-[3px] font-thin">text</div></div><div class="pr-lg"><span><code><span><span>├── requirements.txt
</span></span><span>├── README.md
</span><span>├── notebook.ipynb           # Modular project notebook (all phases)
</span><span>├── cnn_model.h5             # Saved best CNN model (provided separately)
</span><span>├── plant_classifier_RF.pkl  # Saved Random Forest model (provided separately)
</span><span>├── app.py                   # Streamlit deployment script
</span><span>├── utils/                   # (Optional) Custom preprocessing, helpers
</span><span>├── data/                    # (To be created) Place the unzipped PlantVillage data here
</span><span></span></code></span></div></div></div></pre>

## Installation

1. **Clone the repository**
   <pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-textMainDark selection:text-super selection:bg-super/10 bg-offset my-md relative flex flex-col rounded font-mono text-sm font-thin"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl sticky top-0 flex h-0 items-start justify-end"></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-text-200 bg-background-300 py-xs px-sm inline-block rounded-br rounded-tl-[3px] font-thin">text</div></div><div class="pr-lg"><span><code><span><span>git clone https://github.com/yourusername/plant-disease-detection.git
   </span></span><span>cd plant-disease-detection
   </span><span></span></code></span></div></div></div></pre>
2. **Install dependencies**
   <pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-textMainDark selection:text-super selection:bg-super/10 bg-offset my-md relative flex flex-col rounded font-mono text-sm font-thin"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl sticky top-0 flex h-0 items-start justify-end"></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-text-200 bg-background-300 py-xs px-sm inline-block rounded-br rounded-tl-[3px] font-thin">text</div></div><div class="pr-lg"><span><code><span><span>pip install -r requirements.txt
   </span></span><span></span></code></span></div></div></div></pre>
3. **Download and extract the PlantVillage dataset from [here](https://data.mendeley.com/datasets/tywbtsjrjv/1). Place the extracted folders in `./data/` or as specified in the notebook/app.**

## Usage

## 1. Jupyter/Colab Workflow

* Open `notebook.ipynb` in Jupyter or Colab.
* Each phase (feature extraction, random forest, CNN, transfer learning, evaluation, and model export) is a self-contained section.
* Set data paths as needed.
* Run cells sequentially. Models and feature arrays will be saved for reuse.

## 2. Streamlit App

* Place `cnn_model.h5` (and `plant_classifier_RF.pkl` if desired) in the project root.
* Launch the app:
  <pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-textMainDark selection:text-super selection:bg-super/10 bg-offset my-md relative flex flex-col rounded font-mono text-sm font-thin"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl sticky top-0 flex h-0 items-start justify-end"></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-text-200 bg-background-300 py-xs px-sm inline-block rounded-br rounded-tl-[3px] font-thin">text</div></div><div class="pr-lg"><span><code><span><span>streamlit run app.py
  </span></span><span></span></code></span></div></div></div></pre>
* Upload a single leaf image through the UI. The app will display the predicted disease/health class and confidence.

*Demo:*

![Streamlit Demo](Streamlit%20Demo.png)




## Model Details

* **Classical ML:** Random Forest, handcrafted features (`plant_classifier_RF.pkl`)
* **Deep Learning:** Custom CNN with three convolutional blocks, dropout, and dense head (`cnn_model.h5`)
* **(Not deployed)** Transfer Learning (EfficientNetB4): Encountered overfitting to training data, validation accuracy low; not included by design.

## Results and Observations

| Model             | Training Accuracy | Validation Accuracy |
| ----------------- | ----------------- | ------------------- |
| Random Forest     | ~90%              | ~89%                |
| Custom CNN        | ~97%              | ~95%                |
| EfficientNet (TL) | ~97%              | 1–15%              |

* **Best Model Deployed:** Custom-trained CNN (`cnn_model.h5`)
* Extensive pipeline and data debugging ensured the best possible reliability and generalization for production use.

## Acknowledgments

* [PlantVillage dataset creators](https://data.mendeley.com/datasets/tywbtsjrjv/1)
* TensorFlow, Keras, scikit-learn, Streamlit, and the open-source scientific Python community.

## License

This project is open for academic, research, and personal use. Data credit: PlantVillage/Mendeley.
