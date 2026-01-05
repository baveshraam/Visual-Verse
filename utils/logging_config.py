"""
Logging Configuration for VisualVerse
Provides file-based logging for debugging API calls and pipeline execution.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Create logs directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Log file paths
API_LOG = os.path.join(LOG_DIR, 'api.log')
PIPELINE_LOG = os.path.join(LOG_DIR, 'pipeline.log')
ERROR_LOG = os.path.join(LOG_DIR, 'error.log')

# Custom formatter with detailed info
class DetailedFormatter(logging.Formatter):
    def format(self, record):
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return super().format(record)

# Format string
LOG_FORMAT = '%(timestamp)s | %(levelname)-8s | %(name)s | %(message)s'


def setup_file_logging():
    """Configure file-based logging for the application."""
    
    # Create formatters
    detailed_formatter = DetailedFormatter(LOG_FORMAT)
    
    # API Logger - logs all API calls
    api_logger = logging.getLogger('api')
    api_logger.setLevel(logging.DEBUG)
    api_handler = RotatingFileHandler(
        API_LOG, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    api_handler.setFormatter(detailed_formatter)
    api_logger.addHandler(api_handler)
    
    # Pipeline Logger - logs pipeline execution
    pipeline_logger = logging.getLogger('pipeline')
    pipeline_logger.setLevel(logging.DEBUG)
    pipeline_handler = RotatingFileHandler(
        PIPELINE_LOG, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    pipeline_handler.setFormatter(detailed_formatter)
    pipeline_logger.addHandler(pipeline_handler)
    
    # Error Logger - logs all errors
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    error_handler = RotatingFileHandler(
        ERROR_LOG, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    error_handler.setFormatter(detailed_formatter)
    error_logger.addHandler(error_handler)
    
    return api_logger, pipeline_logger, error_logger


def log_api_call(endpoint: str, method: str, data_size: int = 0):
    """Log an API call."""
    api_logger = logging.getLogger('api')
    api_logger.info(f"[{method}] {endpoint} | Data: {data_size} bytes")


def log_api_response(endpoint: str, status: int, duration_ms: float):
    """Log an API response."""
    api_logger = logging.getLogger('api')
    api_logger.info(f"[RESPONSE] {endpoint} | Status: {status} | Duration: {duration_ms:.1f}ms")


def log_pipeline_step(pipeline: str, step: str, message: str):
    """Log a pipeline step."""
    pipeline_logger = logging.getLogger('pipeline')
    pipeline_logger.info(f"[{pipeline}] {step} | {message}")


def log_error(context: str, error: Exception):
    """Log an error."""
    error_logger = logging.getLogger('error')
    error_logger.error(f"[{context}] {type(error).__name__}: {error}")


# Initialize on import
api_logger, pipeline_logger, error_logger = setup_file_logging()

# Write startup marker
api_logger.info("=" * 60)
api_logger.info(f"VisualVerse API Started at {datetime.now().isoformat()}")
api_logger.info("=" * 60)
