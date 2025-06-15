import sys
import os

# Manually add the parent folder of 'app' to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now import the module
from app.matching.semantic_matcher import match_cv_with_jd

def test_semantic_matcher():
    cv_text = """Abuzar Khan is an AI intern with experience in machine learning, data science, and software development.
    Worked with various AI algorithms for data prediction and analysis."""
    job_description = "Looking for a data scientist with machine learning skills and experience in Python."

    # Get matching results
    match_summary = match_cv_with_jd(cv_text, job_description)

    # Check if the match summary contains necessary fields
    assert "best_match_sentence" in match_summary
    assert "best_match_score" in match_summary
    assert match_summary["best_match_score"] > 0.0  # Ensure there's a positive match score
    assert match_summary["skills_similarity"] >= 0.0
    assert match_summary["experience_similarity"] >= 0.0
    assert match_summary["education_similarity"] >= 0.0  # Ensure these values are non-negative

    # Print out the results for inspection
    print(match_summary)

# Run the test
test_semantic_matcher()
