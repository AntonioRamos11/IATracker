import requests
##semantic i need api
def get_semantic_scholar_papers(query="artificial intelligence", num_papers=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={num_papers}"
    headers = {"x-api-key": "YOUR_SEMANTIC_SCHOLAR_API_KEY"}  # Replace with your actual API key
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        papers = [{"title": p["title"], "url": f"https://www.semanticscholar.org/paper/{p['paperId']}"} for p in data["data"]]
        return papers
    else:
        print("Error fetching data:", response.status_code)
        return []

# Example usage
papers = get_semantic_scholar_papers()
for p in papers[:3]:
    print(f"Title: {p['title']}\nURL: {p['url']}\n")