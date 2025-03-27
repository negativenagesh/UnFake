<div>

<div align="center">
    <img src="https://github.com/negativenagesh/UnFake/blob/main/UnFake-logo/logo.png" alt="UnFake Logo" style="width: 700px; height: 300px;">
    <p>Real pics or AI tricks? We’ve got your back before you click!</p>
</div>
</div>

<div align="center">
  
![GitHub stars](https://img.shields.io/github/stars/negativenagesh/deep-fake?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/negativenagesh/deep-fake?style=social)
![GitHub forks](https://img.shields.io/github/forks/negativenagesh/deep-fake?style=social)
![GitHub license](https://img.shields.io/github/license/negativenagesh/deep-fake)
</div>

## Overview

UnFake is the first platform to integrate a deepfake detection tool directly into the image-downloading process. UnFake focuses on building a deepfake image classification system for mainly face and (noise:body and whatever images came during scraping below types of images) images from Unsplash, a platform with billions of images. The goal is to determine whether an image is a deepfake or real on Unsplash website, addressing the growing concern of manipulated media on such platforms. Billions of images flood platforms like Unsplash and Adobe Stock, yet none guarantee whether they’re real or AI-generated deepfakes. This lack of transparency leaves users vulnerable to misinformation, legal liabilities (e.g., copyright or defamation issues), and reputational damage.

Unsplash, a prominent platform for free, high-quality images, has become a critical resource for a diverse user base, including marketers, designers, and educators. However, the rise of deepfake technology, leveraging advanced artificial intelligence and deep learning, poses significant challenges to the authenticity of digital media on such platforms. UnFake focuses on building a deepfake image classification system for Unsplash, particularly for face and body images, and examines the implications for users and the legal system.

Unsplash, as described on its official website [Unsplash](https://unsplash.com/), hosts over 5 million photos and generates more than 13 billion photo impressions monthly, according to its Wikipedia entry [Unsplash - Wikipedia](https://en.wikipedia.org/wiki/Unsplash). It is a go-to resource for users seeking royalty-free images for various purposes, from creative projects to marketing campaigns. The platform's popularity, with over 330,000 contributing photographers, underscores its importance in digital media ecosystems. But Deepfake tech has blurred the lines between reality and fabrication, making it increasingly difficult to distinguish real images from manipulated ones.

## Why UnFake Exists?: Reason to build this
Unsplash, a platform hosting billions of images, faces the challenge of deepfake images, particularly those involving human faces and bodies. These AI-manipulated images pose a significant threat to the authenticity of digital media, impacting users who rely on Unsplash for creative and professional needs. The problem is to develop a deepfake image classification system that can accurately identify whether an image is real or a deepfake, thereby protecting users from potential harms such as misinformation, reputation damage, and legal liabilities.

### Demo (Click on image to watch video):

<div align="center">
  <a href="https://drive.google.com/file/d/1Du6u_1jxerVEeP3ZST4kDoONDGW7i9di/view?usp=sharing">
    <img src="https://imgur.com/a/unfake-PHtJCjV.jpg" alt="Resume review video preview" width="400" style="display:inline-block; margin:10px;">
</div>

## Users of Unsplash and Similar Sites
These users rely on Unsplash for its accessibility and quality, making the presence of deepfake images a significant concern. For instance, marketers using deepfake images in campaigns could face legal repercussions for misleading advertising, while educators might inadvertently use manipulated images in teaching materials, spreading misinformation.

1. Designers and Creatives: For inspiration, mockups, and designs.
2. Marketers and Advertisers: For campaigns, social media, and websites.
3. Small Business Owners: For websites, blogs, and marketing materials.
4. Individuals: For personal projects like collages or memes.
5. Developers: For UI/UX design or placeholders.
6. Educators: For teaching materials and presentations.


## Legal and Judicial Implications
Deepfake images can lead to:

1. Copyright and Ownership Issues: Using someone's likeness without consent may violate publicity or privacy rights.
2. Defamation and Libel: False portrayals can damage reputations, leading to legal action.
3. Misleading Advertising: Using deepfakes in ads can breach advertising laws.
4. Data Protection and Privacy: Creating deepfakes may violate data protection laws like GDPR.
5. Criminal Offenses: In some jurisdictions, deepfakes, especially pornographic or harassing, can be criminal.
6. Contractual Breaches: Uploading manipulated content may violate platform terms, risking penalties.

## Scope of training
Dataset and Categories
  - The dataset comprises around 2,50,000 in which around 76,000 images are scraped from Unsplash using its API and rest from publically available datasets(with deepfake images and some are generated using stable diffusion) and the images are categorized into:

* General Human Faces:
  - Human face images.
* Ethnic Diversity:
  -Asian,Black,Caucasian face images.
* Facial Features:
  -Bearded, freckles, wrinkled, and spectacles face images.
* Age Variation:
  -Child face images.
* Pose & Composition:
  -Close-up and headshot/portrait images.

### System Design
```txt
+-------------------------------------------------------------+
|                        [User] 👤                            |
|   ① Clicks "Get Started"                                   |
|   ② Searches Images, Views Details, Classifies             |
+-------------------------------------------------------------+
              ↓ (HTTP Requests)
+-------------------------------------------------------------+
|                  Frontend Layer (Streamlit App) 🌐          |
|   +-------------------+   +-------------------+             |
|   | Homepage          |-->| Search Page       |             |
|   | - "Get Started"   |   | - Search Bar      |             |
|   +-------------------+   +-------------------+             |
|   +-------------------+   +-------------------+             |
|   | Image Results     |-->| Details Page      |             |
|   | - Unsplash Images |   | - Image Info      |             |
|   | - "View Details"  |   | - "Classify" Btn  |             |
|   +-------------------+   +-------------------+             |
+-------------------------------------------------------------+
              ↓ (API Calls)
+-------------------------------------------------------------+
	|                  Backend Layer ⚙️                         |
|   +-------------------+   +-------------------+             |
|   | /search Endpoint  |   | /classify Endpoint|             |
|   | - Query Unsplash  |   | - Deepfake Check  |             |
|   +-------------------+   +-------------------+             |
|          ↓                        ↓                         |
|   +-------------------+   +-------------------+             |
|   | Unsplash API      |   | Deepfake Model    |             |
|   | - Fetch Images    |   | - Process Image   |             |
|   +-------------------+   +-------------------+             |
|          ↓                        ↓                         |
|   +-------------------+   +-------------------+             |
|   |                   |   | Deepfake Model    |             |
|   | - Store Images    |   | - Classify Image  |             |
|   +-------------------+   +-------------------+             |
|          ↓                        ↓                         |
|   +-------------------+   +-------------------+             |
|   | Image Results     |   | Classification    |             |
|   | - URLs, Metadata  |   | - Real/Deepfake   |             |
|   +-------------------+   +-------------------+             |
+-------------------------------------------------------------+
              ↑ (Responses)
+-------------------------------------------------------------+
|                        [User] 👤                            |
|   Sees Images, Details, and Deepfake Results                |
+-------------------------------------------------------------+

```
### Landing page

<div align="center">
    
![image](https://github.com/user-attachments/assets/1ccafe98-7c28-4b0a-858f-5de4da3a7f0a)
</div>

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
pip install -r pkgs.txt
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

Model Training:

<div align="center">
    <img src="Model/image.png" alt="LAE" style="width: 1000px; height: 600px;">
</div>

<div style=" border-radius: 10px; animation: fadeOutIn 2s infinite;"> <h2 style="color: #00d4ff;">License</h2> <p style="color: #b0b0b3;"> Resumai is licensed under the <a href="https://github.com/negativenagesh/deep-fake/blob/main/LICENSE">Apache License Version 2.0</a>. Feel free to use, modify, and share! ❤️ </p> 
</div>
