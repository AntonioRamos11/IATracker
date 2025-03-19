import fitz  # PyMuPDF
import os
import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

def configure_paths() -> Tuple[Path, Path]:
    """Configure and validate directory paths"""
    pdf_dir = Path("./papers/arxiv/")
    processed_dir = Path("./Database/processed_pdfs/")
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory {pdf_dir} does not exist")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    return pdf_dir, processed_dir

def generate_content_hash(file_path: Path) -> str:
    """Generate SHA-256 hash of file content"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_arxiv_id(file_path: Path) -> Optional[str]:
    """Extract arXiv ID from filename if present"""
    filename = file_path.stem
    if filename.startswith('arxiv-'):
        return filename.split('-')[1]
    return None

def process_pdf(file_path: Path) -> Dict[str, Any]:
    """Process a single PDF file with robust error handling"""
    content_hash = generate_content_hash(file_path)
    output_path = processed_dir / f"{content_hash}.json"
    
    # Skip already processed files
    if output_path.exists():
        logger.info(f"Skipping already processed file: {file_path.name}")
        return {"status": "skipped", "path": file_path, "content_hash": content_hash}

    try:
        with fitz.open(file_path) as doc:
            # Validate PDF structure
            if doc.is_encrypted:
                raise PDFProcessingError("Encrypted PDF not supported")
            
            # Extract text with error handling per page
            text = []
            for page_num, page in enumerate(doc):
                try:
                    text.append(page.get_text())
                except Exception as page_error:
                    logger.warning(f"Error processing page {page_num} in {file_path.name}: {page_error}")
                    continue

            full_text = "\n".join(text)

            raw_date = doc.metadata.get("creationDate")
            if raw_date and raw_date.startswith("D:"):
                # Remove "D:" and optionally the trailing "Z"
                cleaned_date = raw_date[2:].rstrip("Z")
                # Optionally, convert to a proper datetime object and then to ISO format
                from datetime import datetime
                try:
                    dt = datetime.strptime(cleaned_date, "%Y%m%d%H%M%S")
                    published_date = dt.isoformat()
                except ValueError:
                    published_date = cleaned_date
            else:
                published_date = raw_date
                    
            # Enhanced metadata extraction
            metadata = {
                "source_path": str(file_path),
                "content_hash": content_hash,
                "arxiv_id": extract_arxiv_id(file_path),
                "title": doc.metadata.get("title") or file_path.stem,
                "author": doc.metadata.get("author", "Unknown"),
                "creation_date": doc.metadata.get("creationDate"),
                "published": published_date, 
                "pages": len(doc),
                "size": file_path.stat().st_size,
                "file_mtime": file_path.stat().st_mtime,
            }

            # Save results with atomic write
            temp_path = output_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "text": full_text,
                    "metadata": metadata
                }, f, ensure_ascii=False, indent=2)
            
            os.replace(temp_path, output_path)
            return {
                "status": "processed",
                "content_hash": content_hash,
                "title": metadata["title"],
                # ...other metadata...
            }

    except Exception as e:
        logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
        if "Encrypted" in str(e) or "not found" in str(e):  
            quarantine_path = processed_dir / "quarantine" / file_path.name
            quarantine_path.parent.mkdir(exist_ok=True)
            file_path.rename(quarantine_path)
        return {"status": "failed", "path": file_path, "error": str(e)}

def process_pdfs_parallel(pdf_dir: Path, max_workers: int = 4):
    """Process PDFs in parallel with resource limits"""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Initialize counters
    new_processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, f): f for f in pdf_files}
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                if result["status"] == "processed":
                    logger.info(f"Processed {file_path.name} → {result['content_hash']}")
                    new_processed_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1
                else:  # failed
                    failed_count += 1
            except Exception as e:
                logger.error(f"Processing failed for {file_path.name}: {str(e)}")
                failed_count += 1
    
    # Log summary with counts
    logger.info("=" * 50)
    logger.info(f"PDF Processing Summary:")
    logger.info(f"  Total files found:    {len(pdf_files)}")
    logger.info(f"  Newly processed:      {new_processed_count}")
    logger.info(f"  Already processed:    {skipped_count}")
    logger.info(f"  Failed to process:    {failed_count}")
    logger.info("=" * 50)
    
    return new_processed_count, skipped_count, failed_count

if __name__ == "__main__":
    try:
        pdf_dir, processed_dir = configure_paths()
        new_count, skipped_count, failed_count = process_pdfs_parallel(pdf_dir)
        logger.info(f"PDF processing completed. New: {new_count}, Skipped: {skipped_count}, Failed: {failed_count}")
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}", exc_info=True)
        raise SystemExit(1) from e