import logging
from colorama import Fore, Style, init

# Initialize colorama for cross-platform support
init(autoreset=True)

# Define color mapping
COLOR_MAP = {
    'DEBUG': Fore.CYAN + Style.BRIGHT,
    'INFO': Fore.GREEN + Style.BRIGHT,
    'WARNING': Fore.YELLOW + Style.BRIGHT,
    'ERROR': Fore.RED + Style.BRIGHT,
    'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
    'DEFAULT': Fore.WHITE + Style.BRIGHT
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLOR_MAP.get(record.levelname, COLOR_MAP['DEFAULT'])
        # Store original values
        orig_levelname = record.levelname
        orig_msg = record.msg
        
        # Apply color
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        
        # Get formatted string
        result = super().format(record)
        
        # Restore original values
        record.levelname = orig_levelname
        record.msg = orig_msg
        
        return result

def get_custom_logger(name: str = __name__, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Use your original format with the colored formatter
    formatter = ColoredFormatter('%(levelname)s - %(name)s - L:%(lineno)d: %(message)s')
    console_handler.setFormatter(formatter)
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(console_handler)
    
    return logger
