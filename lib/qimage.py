from PyQt5.QtCore import QBuffer, QIODevice
from PyQt5.QtGui import QImage
import base64


def qimage_to_base64(image: QImage, format: str = 'PNG') -> str:
    """
    Convert QImage to base64 string.

    Args:
        image (QImage): The input QImage
        format (str): Image format ('PNG', 'JPG', etc.). Defaults to 'PNG'

    Returns:
        str: Base64 encoded string of the image
    """
    # Check if image is valid
    if image.isNull():
        raise ValueError("Invalid QImage provided")

    # Create a buffer to store the image data
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)

    # Save the image to the buffer in the specified format
    success = image.save(buffer, format)
    if not success:
        raise ValueError(f"Failed to save image in {format} format")

    # Get the buffer data and encode it to base64
    image_data = buffer.data()
    base64_data = base64.b64encode(image_data).decode()

    # Create the complete base64 string with data URI scheme
    mime_type = f"image/{format.lower()}"
    base64_string = f"data:{mime_type};base64,{base64_data}"

    return base64_string

def base64_to_qimage(base64_string: str) -> QImage:
    """
    Convert base64 string back to QImage.

    Args:
        base64_string (str): The base64 encoded image string

    Returns:
        QImage: The decoded image
    """
    # Remove the data URI scheme if present
    if ';base64,' in base64_string:
        base64_string = base64_string.split(';base64,')[1]

    # Decode base64 string to bytes
    image_data = base64.b64decode(base64_string)

    # Create QImage from the decoded data
    image = QImage()
    success = image.loadFromData(image_data)

    if not success:
        raise ValueError("Failed to create QImage from base64 string")

    return image
