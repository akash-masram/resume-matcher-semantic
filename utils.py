import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# Load spaCy model for preprocessing (tokenization, lemmatization)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load Sentence-BERT model once to avoid reloading on every request
model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess(text: str) -> str:
    """
    Clean and lemmatize input text using spaCy.
    Removes stopwords and non-alphabetic tokens.
    Returns a space-separated string of processed tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
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
