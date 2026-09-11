"""Enhanced endpoint helpers for better error handling and recovery."""
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException

from app.services.cv.latex_generator_enhanced import (
    EnhancedLaTeXCompiler,
    LaTeXTimeoutError,
    LaTeXCompilationFailed,
)
from app.services.parsing.parser_fallback import RobustResumeParser

logger = logging.getLogger(__name__)


class EnhancedEndpointHandlers:
    """Enhanced handlers with better error recovery."""
    
    def __init__(self):
        self.latex_compiler = EnhancedLaTeXCompiler(
            timeout_seconds=30,
            max_retries=2,
            enable_diagnostics=True,
        )
        self.resume_parser = RobustResumeParser()
    
    async def extract_resume_text_with_fallback(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> str:
        """Extract resume text with multiple fallback strategies.
        
        Raises:
            HTTPException: If all extraction strategies fail
        """
        # Determine file type from filename
        file_type = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'pdf'
        
        text, metadata = self.resume_parser.extract_from_file(file_bytes, file_type)
        
        if not text:
            logger.error(
                f"Resume extraction failed. Attempted: {metadata.get('strategies_attempted')}. "
                f"Errors: {metadata.get('errors')}"
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not extract text from resume. "
                    f"Tried: {', '.join(metadata.get('strategies_attempted', []))}. "
                    f"Please ensure the file is a valid {file_type.upper()} document."
                ),
            )
        
        logger.info(
            f"Successfully extracted resume using {metadata.get('strategy_used')}: "
            f"{metadata.get('text_length')} characters"
        )
        
        return text
    
    async def compile_latex_with_recovery(
        self,
        latex_content: str,
        output_name: str = "cv",
    ) -> bytes:
        """Compile LaTeX to PDF with timeout and recovery.
        
        Raises:
            HTTPException: If compilation ultimately fails
        """
        try:
            pdf_bytes = self.latex_compiler.compile_latex_to_pdf(
                latex_content,
                output_name,
            )
            return pdf_bytes
        
        except LaTeXTimeoutError as e:
            logger.error(f"LaTeX compilation timeout: {e}")
            raise HTTPException(
                status_code=504,
                detail=(
                    f"LaTeX PDF compilation timed out after "
                    f"{self.latex_compiler.timeout_seconds} seconds. "
                    f"Your CV may be too complex. Try simplifying it or contact support."
                ),
            )
        
        except LaTeXCompilationFailed as e:
            logger.error(f"LaTeX compilation failed: {e}")
            
            error_log = self.latex_compiler.get_last_error_log()
            error_detail = str(e)
            if error_log:
                error_detail += f"\n\nDiagnostics:\n{error_log}"
            
            raise HTTPException(
                status_code=500,
                detail=(
                    f"LaTeX PDF generation failed. This usually means there's a syntax error "
                    f"in the generated CV template. Technical details: {error_detail[:200]}"
                ),
            )
