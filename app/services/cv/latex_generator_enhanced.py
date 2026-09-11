"""Enhanced LaTeX compiler with timeout and recovery mechanisms."""
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import shutil

logger = logging.getLogger(__name__)


class LaTeXCompilationError(Exception):
    """Base class for LaTeX compilation errors."""
    pass


class LaTeXTimeoutError(LaTeXCompilationError):
    """Raised when LaTeX compilation times out."""
    pass


class LaTeXCompilationFailed(LaTeXCompilationError):
    """Raised when LaTeX compilation fails."""
    pass


class EnhancedLaTeXCompiler:
    """Enhanced LaTeX compiler with timeout, error recovery, and diagnostics."""
    
    def __init__(
        self,
        timeout_seconds: int = 30,
        max_retries: int = 2,
        enable_diagnostics: bool = True,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.enable_diagnostics = enable_diagnostics
        self.last_error_log = None
    
    def compile_latex_to_pdf(
        self,
        latex_content: str,
        output_name: str = "output",
    ) -> bytes:
        """Compile LaTeX to PDF with timeout and recovery.
        
        Args:
            latex_content: Complete LaTeX document source
            output_name: Name for output PDF (without extension)
        
        Returns:
            PDF file bytes
        
        Raises:
            LaTeXTimeoutError: If compilation exceeds timeout
            LaTeXCompilationFailed: If compilation fails
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tex_file = tmp_path / f"{output_name}.tex"
            pdf_file = tmp_path / f"{output_name}.pdf"
            
            # Write LaTeX source
            tex_file.write_text(latex_content, encoding='utf-8')
            
            # Attempt compilation with retries
            for attempt in range(self.max_retries + 1):
                try:
                    logger.info(
                        f"LaTeX compilation attempt {attempt + 1}/{self.max_retries + 1}"
                    )
                    
                    self._run_pdflatex(
                        tex_file,
                        tmp_path,
                        output_name,
                    )
                    
                    # Check if PDF was generated
                    if pdf_file.exists():
                        logger.info(f"PDF compiled successfully: {pdf_file.stat().st_size} bytes")
                        return pdf_file.read_bytes()
                    else:
                        raise LaTeXCompilationFailed(
                            f"pdflatex completed but no PDF generated at {pdf_file}"
                        )
                
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"LaTeX compilation timed out after {self.timeout_seconds}s "
                        f"(attempt {attempt + 1})"
                    )
                    if attempt >= self.max_retries:
                        raise LaTeXTimeoutError(
                            f"LaTeX compilation timed out after {self.max_retries} retries. "
                            f"Timeout: {self.timeout_seconds}s"
                        )
                
                except LaTeXCompilationFailed as e:
                    logger.warning(f"LaTeX compilation failed: {e}")
                    
                    # Extract diagnostics
                    if self.enable_diagnostics:
                        self._extract_diagnostics(tmp_path, output_name)
                    
                    if attempt >= self.max_retries:
                        raise
            
            raise LaTeXCompilationFailed(
                "LaTeX compilation failed after all retries"
            )
    
    def _run_pdflatex(
        self,
        tex_file: Path,
        work_dir: Path,
        output_name: str,
    ) -> None:
        """Run pdflatex with timeout.
        
        Raises:
            subprocess.TimeoutExpired: If compilation times out
            LaTeXCompilationFailed: If pdflatex returns error
        """
        try:
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={work_dir}",
                    str(tex_file),
                ],
                cwd=work_dir,
                timeout=self.timeout_seconds,
                capture_output=True,
                text=True,
            )
            
            # Log output for debugging
            if result.returncode != 0:
                logger.warning(f"pdflatex stderr: {result.stderr[:500]}")
                logger.warning(f"pdflatex stdout: {result.stdout[:500]}")
                
                raise LaTeXCompilationFailed(
                    f"pdflatex exited with code {result.returncode}"
                )
        
        except subprocess.TimeoutExpired:
            raise
        except Exception as e:
            raise LaTeXCompilationFailed(str(e))
    
    def _extract_diagnostics(self, work_dir: Path, output_name: str) -> None:
        """Extract error diagnostics from LaTeX log file."""
        log_file = work_dir / f"{output_name}.log"
        
        if not log_file.exists():
            logger.warning("No LaTeX log file found")
            return
        
        try:
            log_content = log_file.read_text(encoding='utf-8', errors='ignore')
            
            # Extract error lines
            errors = []
            for line in log_content.split('\n'):
                if any(marker in line for marker in ['Error', 'error', '!', 'Missing']):
                    errors.append(line.strip())
            
            self.last_error_log = '\n'.join(errors[:10])  # Keep first 10 errors
            
            if errors:
                logger.error(f"LaTeX errors:\n{self.last_error_log}")
        
        except Exception as e:
            logger.warning(f"Could not extract LaTeX diagnostics: {e}")
    
    def get_last_error_log(self) -> Optional[str]:
        """Get the last error log from LaTeX compilation."""
        return self.last_error_log
