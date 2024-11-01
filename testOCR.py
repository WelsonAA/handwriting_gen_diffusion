import pytesseract
from PIL import Image


def perform_ocr(img_path, expected_text):
    # Load the image
    img = Image.open(img_path)

    # Perform OCR on the image
    recognized_text = pytesseract.image_to_string(img, config='--psm 7')

    # Print the results
    print("Recognized Text from Image:")
    print(recognized_text.strip())
    print("\nExpected Text:")
    print(expected_text)

    # Check if the recognized text matches the expected text (basic comparison)
    if recognized_text.strip() == expected_text:
        print("\nMatch Status: The recognized text matches the expected text.")
    else:
        print("\nMatch Status: The recognized text does NOT match the expected text.")


# Example usage
img_path = "./output/sample.png"  # Replace with your image file path
expected_text = "I love Diffusion"
perform_ocr(img_path, expected_text)
