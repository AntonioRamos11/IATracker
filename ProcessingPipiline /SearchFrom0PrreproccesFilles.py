import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_empty_or_zero_files(processed_dir: Path):
    """Search for preprocessed files that contain '0', are empty, or have the shortest length"""
    shortest_length = float('inf')
    shortest_files = []

    for json_file in processed_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Check if the text is "0" or empty
                text = data.get("text", "").strip()
                if text == "0" or not text:
                    logger.info(f"Found file with '0' or empty content: {json_file}")
                
                # Compare the length of the text
                text_length = len(text)
                if text_length < shortest_length:
                    shortest_length = text_l    ength
                    shortest_files = [json_file]
                elif text_length == shortest_length:
                    shortest_files.append(json_file)
        
        except Exception as e:
            logger.error(f"Failed to read {json_file}: {e}", exc_info=True)

    # Log the files with the shortest length
    for file in shortest_files:
        logger.info(f"File with shortest length ({shortest_length} characters): {file}")

if __name__ == "__main__":
    processed_dir = Path("./Database/processed_pdfs/")
    if not processed_dir.exists():
        logger.error(f"Processed directory {processed_dir} does not exist")
        raise SystemExit(1)
    
    search_empty_or_zero_files(processed_dir)
    print("Search completed")