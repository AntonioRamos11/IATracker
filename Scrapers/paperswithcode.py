import requests

def get_papers_with_code(query="artificial intelligence", num_papers=5):
    url = f"https://paperswithcode.com/api/v1/papers/?q={query}&items_per_page={num_papers}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        papers = [
            {
                "title": p.get("title"),
                "url": p.get("url"),
                "abstract": p.get("abstract"),
                "pdf_url": p.get("url_pdf")
            }
            for p in data.get("results", [])
        ]
        return papers
    else:
        print("Error fetching data:", response.status_code)
        return []

# Example usage
papers = get_papers_with_code()
for p in papers[:3]:
    print(f"Title: {p['title']}\nLink: {p['url']}\nPDF: {p['pdf_url']}\n")