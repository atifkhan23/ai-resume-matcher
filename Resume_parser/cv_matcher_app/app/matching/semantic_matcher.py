import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Add this line to avoid GUI issues
import matplotlib.pyplot as plt
import io
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud


class SemanticScorer:
    """
    Computes semantic similarity between CV and Job Description sections (skills, experience, education).
    Generates missing-keyword word clouds and SHAP-based explainability charts.
    """

    SECTIONS = ['skills', 'experience', 'education']

    def __init__(self):
        # Load a lightweight sentence-transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Section weights must sum to 1.0
        self.weights = {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.15,
            'others': 0.15
        }
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("The weights must sum to 1.0")

    def embed(self, text: str):
        """
        Generate a normalized sentence embedding for the given text.
        """
        text = text.strip() if text else ""
        return self.model.encode([text], normalize_embeddings=True)

    def get_similarity(self, emb1, emb2) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        Returns a float between -1 and 1.
        """
        return float(cosine_similarity(emb1, emb2)[0][0])

    def score_cv(self, cv: dict, jd: dict) -> dict:
        """
        Score each section of the CV against the JD and return detailed metrics.
        Returns a dict with section-wise similarities, best match, and total weighted score.
        """
        section_scores = {}
        total_score = 0.0
        best_match_score = -1.0
        best_match_section = ''

        # Precompute embeddings for CV and JD sections
        cv_embeds = {sec: self.embed(cv.get(sec, '')) for sec in self.SECTIONS}
        jd_embeds = {sec: self.embed(jd.get(sec, '')) for sec in self.SECTIONS}

        # Compute similarity per section
        for sec in self.SECTIONS:
            sim = self.get_similarity(cv_embeds[sec], jd_embeds[sec])
            section_scores[f"{sec}_similarity"] = sim
            weighted_score = sim * 100 * self.weights.get(sec, 0)
            total_score += weighted_score
            if sim > best_match_score:
                best_match_score = sim
                best_match_section = sec

        # Aggregate results
        section_scores.update({
            'best_match_score': round(best_match_score * 100, 2),
            'best_match_section': best_match_section.capitalize(),
            'total_score': round(total_score, 2)
        })
        return section_scores

    def find_missing_keywords(self, cv: dict, jd: dict) -> list[str]:
        """
        Identify keywords present in JD sections but missing from CV sections.
        Returns a sorted list of missing words.
        """
        missing = set()
        for sec in self.SECTIONS:
            cv_words = set(cv.get(sec, "").lower().replace(',', ' ').split())
            jd_words = set(jd.get(sec, "").lower().replace(',', ' ').split())
            missing |= (jd_words - cv_words)
        return sorted(missing)

    def generate_word_cloud(self, keywords: list[str]) -> str | None:
        """
        Create a word cloud PNG (base64-encoded) from a list of keywords.
        Returns None if no keywords provided.
        """
        if not keywords:
            return None

        text = ' '.join(keywords)
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)

        buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def explain_score(self, cv: dict, jd: dict) -> str:
        """
        Use SHAP to explain how each section contributed to the overall score.
        Returns a base64-encoded PNG of a SHAP bar chart with real section names.
        """
        # Compute raw similarities
        sims = [
            self.get_similarity(self.embed(cv.get(sec, '')), self.embed(jd.get(sec, '')))
            for sec in self.SECTIONS
        ]
        X = np.array(sims).reshape(1, -1)

        # Define a dummy model that returns the weighted sum
        def model_predict(data: np.ndarray) -> np.ndarray:
            w = np.array([self.weights[sec] for sec in self.SECTIONS])
            return np.sum(data * w, axis=1).reshape(-1, 1)

        # Explain with SHAP, passing feature names for clarity
        explainer = shap.Explainer(model_predict, np.eye(len(self.SECTIONS)), feature_names=self.SECTIONS)
        shap_values = explainer(X)

        buf = io.BytesIO()
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
