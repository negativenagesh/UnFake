<div>

<h1 align='center'>
  UnFake
</h1>
<p align="center">No Fakes, Just Facts</p>
</div>

<div align="center">
  
![GitHub stars](https://img.shields.io/github/stars/negativenagesh/deep-fake?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/negativenagesh/deep-fake?style=social)
![GitHub forks](https://img.shields.io/github/forks/negativenagesh/deep-fake?style=social)
![GitHub license](https://img.shields.io/github/license/negativenagesh/deep-fake)
</div>

## Project Overview
This project focuses on building a deepfake image classification system for face images from Unsplash, a platform with billions of images. The goal is to determine whether an image is a deepfake or real, addressing the growing concern of manipulated media on such platforms.

The advent of deepfake technology, powered by advances in artificial intelligence and deep learning, has blurred the lines between reality and fabrication in digital media. Deepfakes, particularly image-based, pose significant threats to the authenticity of content on platforms like Unsplash, which hosts billions of images used for creative, professional, and personal purposes. The ability to create highly realistic manipulated images has implications for misinformation, trust, and legal integrity, necessitating robust detection systems. This project addresses this need by developing a classification system to identify whether Unsplash face images are deepfake or real, contributing to the broader effort to safeguard digital media authenticity.

## Scope of training
* Dataset and Categories
  - The dataset comprises around 70,000 images scraped from Unsplash using its API, categorized into:

* General Human Faces: Human face images.
* Ethnic Diversity: Asian, Black, and Caucasian face images.
* Facial Features: Bearded, freckles, wrinkled, and spectacles face images.
* Age Variation: Child face images.
* Pose & Composition: Close-up and headshot/portrait images.

## ⚙️ Setup
1. Star and Fork this repo by clicking 'fork' in right side above, then follow below steps

2. Clone this repo:

```bash
git clone https://github.com/negativenagesh/deep-fake.git 
```
3. Create a venv:
```bash
python3.12 -m venv deep
```
4. Activate venv:
```bash
  source deep/bin/activate
```
5. Install dependencies:
```bash
pip install -r required.txt
```
6. API set-up:
```txt
reate a new file named .env in the root directory of the project:
Get your free API key from here by sending an application:
https://unsplash.com/oauth/applications
```

## 🌐 Unsplash API Overview:

Basic Structure of the Unsplash API
The Unsplash API follows RESTful principles and uses HTTP methods (GET, POST, etc.) to interact with resources. Key features include:

1. Authentication: Requires an access_key (Client ID) for authorization.
2. Endpoints: Various endpoints for searching, downloading, and managing images.
3. Parameters: Query parameters like query, page, per_page, etc., to customize requests.
4. Rate Limiting: Limits the number of requests per hour (e.g., 50 requests per hour for free tier).

# Example Usage

1. Setup and Authentication

```python
import requests
import os
import time

access_key = "YOUR-API-KEY"
url = "https://api.unsplash.com/search/photos"
headers = {
    "Authorization": f"Client-ID {access_key}"
}
params = {
    "query": "Spectacles face", #EXAMPLE OF IMAGE 
    "per_page": 30
}
```
* The access_key is your Unsplash API Client ID.
* The url is the endpoint for searching photos.
* The headers include the authorization token.
* The params dictionary specifies the search query and the number of images per page.

<div style=" border-radius: 10px; animation: fadeOutIn 2s infinite;"> <h2 style="color: #00d4ff;">License</h2> <p style="color: #b0b0b3;"> Resumai is licensed under the <a href="https://github.com/negativenagesh/deep-fake/blob/main/LICENSE">Apache License Version 2.0</a>. Feel free to use, modify, and share! ❤️ </p> 
</div>
