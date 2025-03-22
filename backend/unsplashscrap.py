import requests
import urllib.parse

def scrape_unsplash_images(search_term, page=1, per_page=30):
    """
    Fetch images from Unsplash API
    
    Args:
        search_term (str): The search query
        page (int): Page number
        per_page (int): Number of images per page (max 30)
    
    Returns:
        list: List of dictionaries containing image data
        int: Total number of pages
    """
    url = "https://unsplash.com/napi/search/photos"
    params = {
        "page": page,
        "per_page": min(per_page, 30),  # Ensure per_page does not exceed 30
        "query": urllib.parse.quote(search_term.strip())
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        images = []
        for result in data.get("results", []):
            image = {
                "url": result["urls"].get("full", result["urls"]["regular"]),  # High-resolution URL
                "alt_text": result.get("alt_description", "Unsplash Image"),
                "author": result["user"].get("name", "Unknown"),
                "download_url": result["urls"].get("full", result["urls"]["regular"])  # Same as url for simplicity
            }
            images.append(image)
        return images, data.get("total_pages", 1)
    except Exception as e:
        print(f"Error fetching images: {e}")
        return [], 1    