"""Enhanced resume parser with multiple extraction strategies and fallbacks."""
import logging
from io import BytesIO
from typing import Optional, Tuple

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfFileReader as PdfReader

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from docx import Document

logger = logging.getLogger(__name__)


class PDFExtractionStrategy:
    """Base class for PDF text extraction strategies."""
    
    def extract(self, pdf_bytes: bytes) -> Optional[str]:
        raise NotImplementedError
    
    def name(self) -> str:
        raise NotImplementedError


class PyPDFStrategy(PDFExtractionStrategy):
    """Extract text using pypdf library."""
    
    def extract(self, pdf_bytes: bytes) -> Optional[str]:
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n".join(text_parts) if text_parts else None
        except Exception as e:
            logger.warning(f"PyPDF extraction failed: {e}")
            return None
    
    def name(self) -> str:
        return "PyPDF"


class PDFPlumberStrategy(PDFExtractionStrategy):
    """Extract text using pdfplumber library (fallback)."""
    
    def extract(self, pdf_bytes: bytes) -> Optional[str]:
        if not HAS_PDFPLUMBER:
            return None
        
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                return "\n".join(text_parts) if text_parts else None
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return None
    
    def name(self) -> str:
        return "pdfplumber"


class DocxExtractionStrategy(PDFExtractionStrategy):
    """Extract text from DOCX files."""
    
    def extract(self, docx_bytes: bytes) -> Optional[str]:
        try:
            doc = Document(BytesIO(docx_bytes))
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n".join(text_parts) if text_parts else None
        except Exception as e:
            logger.warning(f"DOCX extraction failed: {e}")
            return None
    
    def name(self) -> str:
        return "DOCX"


class TextExtractionStrategy(PDFExtractionStrategy):
    """Extract text from plain text files."""
    
    def extract(self, text_bytes: bytes) -> Optional[str]:
        try:
            return text_bytes.decode('utf-8').strip()
        except UnicodeDecodeError:
            # Try with latin-1 as fallback encoding
            try:
                return text_bytes.decode('latin-1').strip()
            except Exception as e:
                logger.warning(f"Text extraction failed: {e}")
                return None
    
    def name(self) -> str:
        return "PlainText"


class RobustResumeParser:
    """Resume parser with multiple extraction strategies and fallbacks."""
    
    def __init__(self):
        # Order matters: try most reliable first
        self.pdf_strategies = [
            PyPDFStrategy(),
            PDFPlumberStrategy(),
        ]
        self.strategies = {
            'pdf': self.pdf_strategies,
            'docx': [DocxExtractionStrategy()],
            'txt': [TextExtractionStrategy()],
        }
    
    def extract_from_file(
        self,
        file_bytes: bytes,
        file_type: str,
    ) -> Tuple[Optional[str], dict]:
        """Extract text from file with fallback strategies.
        
        Args:
            file_bytes: Raw file bytes
            file_type: File extension (pdf, docx, txt)
        
        Returns:
            Tuple of (extracted_text, metadata)
            metadata includes:
                - strategy_used: Name of successful strategy
                - extraction_successful: Boolean
                - strategies_attempted: List of attempted strategies
                - errors: List of errors encountered
        """
        file_type = file_type.lower().strip('.')
        
        strategies = self.strategies.get(file_type, [])
        if not strategies:
            return None, {
                'extraction_successful': False,
                'error': f'Unsupported file type: {file_type}',
                'strategies_attempted': [],
            }
        
        errors = []
        strategies_attempted = []
        
        for strategy in strategies:
            strategy_name = strategy.name()
            strategies_attempted.append(strategy_name)
            
            try:
                logger.info(f"Attempting extraction with {strategy_name}...")
                text = strategy.extract(file_bytes)
                
                if text and len(text.strip()) > 50:  # Minimum text threshold
                    logger.info(
                        f"Successfully extracted {len(text)} chars using {strategy_name}"
                    )
                    return text, {
                        'extraction_successful': True,
                        'strategy_used': strategy_name,
                        'text_length': len(text),
                        'strategies_attempted': strategies_attempted,
                        'errors': errors,
                    }
            
            except Exception as e:
                error_msg = f"{strategy_name} failed: {str(e)}"
                errors.append(error_msg)
                logger.debug(error_msg)
        
        # All strategies failed
        return None, {
            'extraction_successful': False,
            'strategies_attempted': strategies_attempted,
            'errors': errors,
            'error': f'All {len(strategies)} extraction strategies failed',
        }
