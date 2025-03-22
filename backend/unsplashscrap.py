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
        "per_page": min(per_page, 30),
        "query": urllib.parse.quote(search_term.strip())
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        images = []
        for result in data.get("results", []):
            image = {
                "id": result.get("id", "unknown"),
                "display_url": result["urls"].get("regular", result["urls"]["small"]),
                "download_url": result["urls"].get("full", result["urls"]["regular"]),
                "alt_text": result.get("alt_description", result.get("description", "Unsplash Image")),
                "author_name": result["user"].get("name", "Unknown"),
                "author_username": result["user"].get("username", "unknown"),
                "height": result.get("height", 0),
                "width": result.get("width", 0),
                "created_at": result.get("created_at", "Unknown"),
                "likes": result.get("likes", 0),
                "color": result.get("color", "#000000")
            }
            images.append(image)
        return images, data.get("total_pages", 1)
    except Exception as e:
        print(f"Error fetching images: {e}")
        return [], 1