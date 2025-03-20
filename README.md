<h1 align='center'>
  Deep fake image classification
</h1>

<div align="center">
  
![GitHub stars](https://img.shields.io/github/stars/negativenagesh/deep-fake?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/negativenagesh/deep-fake?style=social)
![GitHub forks](https://img.shields.io/github/forks/negativenagesh/deep-fake?style=social)
![GitHub license](https://img.shields.io/github/license/negativenagesh/deep-fake)
</div>

# 📚 Project Overview
This project focuses on the classification of deep fake images using deep learning techniques. Deep fake images are artificially generated images that can be used to spread misinformation or create misleading content. By accurately classifying these images, this project aims to contribute to the detection and prevention of such malicious activities. This tool is useful for researchers, security experts, and anyone interested in understanding and combating the spread of deep fake content.

# ⚙️ Setup
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

# 🌐 Unsplash API Overview:

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

# Scope of Training

I have used different types of facial images:
1. General Human Faces:
   1.1. human-face-images

2. Ethnic Diversity:
   2.1. asian-face-images
   2.2. blackface-images
   2.3. caucasianfaces-images

3. Facial Features:
   3.1. beardedface-images
   3.2. frecklesface-images
   3.3. wrinkledface-images
   3.4. spectacles-images

4. Age Variation:
   4.1. childface-images

5. Pose & Composition:
   5.1. closeup-face-images
   5.2. headshot+portrait-images
