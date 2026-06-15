import os
import time
import logging
import threading
from django.core.mail import EmailMessage
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger('security_app')
err_logger = logging.getLogger('error')

# Alert cooldown cache: {track_id: timestamp}
_alert_cooldowns = {}

def send_alert_email_async(track_id: int, camera_name: str, screenshot_path: str, confidence: float) -> bool:
    """
    Checks email cooldown and spawns a background thread to send the security alert email with screenshot.
    """
    now = time.time()
    cooldown = getattr(settings, 'EMAIL_COOLDOWN_SECONDS', 120)
    
    if track_id in _alert_cooldowns:
        last_sent = _alert_cooldowns[track_id]
        if now - last_sent < cooldown:
            logger.info(f"Skipping alert email for Track {track_id} (under cooldown).")
            return False
            
    _alert_cooldowns[track_id] = now
    
    # Spawn the background worker thread
    thread = threading.Thread(
        target=_send_alert_worker,
        args=(track_id, camera_name, screenshot_path, confidence),
        daemon=True
    )
    thread.start()
    return True

def _send_alert_worker(track_id: int, camera_name: str, screenshot_path: str, confidence: float):
    """
    Worker thread task that handles the email packaging, file attachments, and SMTP transmission with retries.
    """
    recipient = getattr(settings, 'ALERT_RECIPIENT', '')
    if not recipient:
        err_logger.error("No ALERT_RECIPIENT configured in settings. Email alert aborted.")
        return
        
    subject = f"SECURITY ALERT: Person Detected on {camera_name}"
    detected_at_str = timezone.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    body = (
        f"SECURITY ALERT\n"
        f"=====================================\n\n"
        f"An unauthorized person was detected by the surveillance system.\n\n"
        f"Details:\n"
        f"  - Camera Name: {camera_name}\n"
        f"  - Track ID: {track_id}\n"
        f"  - Bounding Box Confidence: {confidence:.2%}\n"
        f"  - Timestamp: {detected_at_str}\n\n"
        f"Please view the attached screenshot from the detection event.\n"
    )
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            email = EmailMessage(
                subject=subject,
                body=body,
                from_email=getattr(settings, 'EMAIL_HOST_USER', 'alerts@surveillance-system.com'),
                to=[recipient]
            )
            
            # Attach screenshot if valid
            if screenshot_path and os.path.exists(screenshot_path):
                email.attach_file(screenshot_path)
            else:
                logger.warning(f"Screenshot file missing at: {screenshot_path}. Sending email without attachment.")
                
            email.send(fail_silently=False)
            logger.info(f"Alert email sent successfully for Track {track_id} on attempt {attempt}.")
            break
        except Exception as e:
            err_logger.error(f"Error sending email alert for Track {track_id} on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                err_logger.error(f"Failed to send email alert for Track {track_id} after {max_retries} attempts.")
