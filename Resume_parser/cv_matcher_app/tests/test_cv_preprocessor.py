import sys
import os
import unittest
import base64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.parser.cv_preprocessor import preprocess_cv
from app.parser.cv_structured_parser import extract_structured_fields, generate_experience_image

# Path to the real-time resume PDF
cv_path = r"C:\Resume_parser\Resume(11).pdf"

class TestCVProcessing(unittest.TestCase):

    def test_cv_preprocessing(self):
        # Step 1: Preprocess CV to clean the text
        result = preprocess_cv(cv_path)

        print("=== CLEANED TEXT PREVIEW ===")
        print(result["cleaned_text"][:500])
        print("\nSentence Count:", len(result["sentences"]))

        # Step 2: Extract structured fields
        print("\n=== STRUCTURED DATA ===")
        structured = extract_structured_fields(result["cleaned_text"])

        print("\nCONTACT:")
        for key, value in structured["contact"].items():
            print(f"{key}: {value}")

        print("\nEDUCATION:")
        print(structured["education"])

        print("\nEXPERIENCE:")
        print(structured["experience"])

        print("\nSKILLS:")
        print(structured["skills"])

        print("\nPROJECTS:")
        print(structured["projects"])

        # Verify that structured data contains key sections (example)
        self.assertIn("contact", structured, "Contact information is missing.")
        self.assertIn("experience", structured, "Experience section is missing.")
        self.assertIn("education", structured, "Education section is missing.")
        self.assertIn("skills", structured, "Skills section is missing.")

        # Step 3: Generate Experience Plot (Timeline)
        experience_data = structured["experience"]  # Assuming structured experience data is available
        plot_base64 = generate_experience_image(experience_data)

        # Check if the base64 string for the plot is not empty
        self.assertIsNotNone(plot_base64, "Experience timeline plot was not generated.")
        self.assertGreater(len(plot_base64), 0, "Experience timeline plot is empty (failed to generate).")
        print("\nExperience Timeline Plot Generated Successfully!")

if __name__ == '__main__':
    unittest.main()
