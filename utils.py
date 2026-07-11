import csv
import os
from datetime import datetime


def initialize_csv(log_file):
    """Create CSV file with header if it doesn't exist."""
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Object", "Confidence"])


def log_detection(log_file, object_name, confidence):
    """Append one detection to the CSV file."""
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            object_name,
            round(confidence, 2)
        ])