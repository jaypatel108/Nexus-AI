from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_seasurf import SeaSurf

# Initialize the rate limiter to prevent abuse.
# Limits are based on the user's IP address.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"]
)

# Initialize SeaSurf for CSRF protection.
csrf = SeaSurf()