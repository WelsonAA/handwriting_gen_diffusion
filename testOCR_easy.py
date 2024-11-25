import easyocr

def perform_ocr_with_easyocr(img_path, expected_text):
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])  # Specify English as the language

    # Perform OCR on the image
    results = reader.readtext(img_path)

    # Extract the recognized text
    recognized_text = " ".join([result[1] for result in results])

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
perform_ocr_with_easyocr(img_path, expected_text)
