import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# Load spaCy model for preprocessing (tokenization, lemmatization)
nlp = spacy.blank("en")

# Load Sentence-BERT model once to avoid reloading on every request
model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess(text: str) -> str:
    """
    Simple preprocessing using spaCy blank model.
    Tokenizes text, lowercases, removes stopwords and punctuation.
    """
    if not text or not isinstance(text, str):
        return ""

    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and token.text not in STOP_WORDS]
    return " ".join(tokens)



def get_semantic_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute semantic similarity between two texts using Sentence-BERT.
    Returns a percentage score (0–100).
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    embeddings = model.encode(
        [resume_text, jd_text],
        convert_to_tensor=True,
        normalize_embeddings=True  # Important for cosine similarity
    )
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity[0][0]) * 100, 2)


def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Extract top-N keywords from text using TF-IDF vectorization.
    Returns a list of important keywords sorted by weight.
    """
    if not text.strip():
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    keywords = sorted(zip(feature_array, scores), key=lambda x: x[1], reverse=True)
    return [word for word, _ in keywords[:top_n]]
