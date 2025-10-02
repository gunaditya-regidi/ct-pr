
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import re
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline
import spacy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gtts import gTTS
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

nltk.download('stopwords')

app = Flask(__name__)


summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)  # Use CPU
nlp = spacy.load("en_core_web_trf")

# File paths
data_path = 'data/topic_modelled_clinical_trials.csv'
cluster_labels_path = 'data/cluster_labels.csv'
cluster_centroids_path = 'data/cluster_centroids.csv'

# Load data
data = pd.read_csv(data_path)

def compute_similarity(topic, data, vectorizer, threshold=0.5):
    """Compute the similarity between the topic and the trials."""
    # Preprocess the topic text
    topic = preprocess_text(topic)
    
    # Transform the topic into the feature space
    topic_vector = vectorizer.transform([topic])
    
    # Transform the trials' cleaned_text into the feature space
    trials_vectors = vectorizer.transform(data['cleaned_text'].fillna(''))  # Handle NaN values
    
    # Compute cosine similarity between the topic vector and trials vectors
    similarity_scores = cosine_similarity(topic_vector, trials_vectors).flatten()
    
    # Add similarity scores as a new column to the DataFrame
    data['similarity'] = similarity_scores
    
    # Filter trials based on a similarity threshold
    similar_trials = data[data['similarity'] >= threshold]
    
    return similar_trials



def preprocess_text(text):
    """Preprocess the input text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

def preprocess_conditions(conditions):
    # Normalize and clean conditions
    conditions = conditions.astype(str)
    conditions = conditions.str.lower().str.strip()
    conditions = conditions.apply(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)
    return conditions

def preprocess_and_cluster(data_path, n_clusters=5):
    """Preprocess the data and apply clustering."""
    # Load and preprocess the data
    data = pd.read_csv(data_path)
    data['Conditions'] = preprocess_conditions(data['Conditions'])
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['Conditions'])
    
    # Apply PCA to reduce dimensionality if needed
    pca = None
    if X.shape[1] > 1:  # Apply PCA if more than one feature
        pca = PCA(n_components=min(2, X.shape[1]))  # Limit to the number of features
        X = pca.fit_transform(X.toarray())
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['cluster'] = kmeans.fit_predict(X)
    
    return data, kmeans, vectorizer, pca


def clean_text(text):
    # Remove excessive punctuation and whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'[,;:]+', ', ', text)
    return text

def prepare_summary_data(trials):
    text = ' '.join(trials['Brief Summary'].dropna())
    cleaned_text = clean_text(text)
    return cleaned_text


def generate_structured_summary(trials):
    num_trials = len(trials)
    
    start_dates = pd.to_datetime(trials['Start Date'], errors='coerce').dropna()
    completion_dates = pd.to_datetime(trials['Completion Date'], errors='coerce').dropna()
    
    if not start_dates.empty:
        start_year_range = f"{start_dates.min().year} to {start_dates.max().year}"
    else:
        start_year_range = "No valid start dates available"
        
    if not completion_dates.empty:
        completion_year_range = f"{completion_dates.min().year} to {completion_dates.max().year}"
    else:
        completion_year_range = "No valid completion dates available"
    
    num_interventions = trials['Interventions'].dropna().nunique()
    
    structured_summary = (
        f"Number of trials found: {num_trials}\n"
        f"Trial Start Date Range: {start_year_range}\n"
        f"Trial Completion Date Range: {completion_year_range}\n"
        f"Number of unique interventions: {num_interventions}\n"
    )
    
    return structured_summary






def get_related_trials(topic, data, kmeans, vectorizer, pca=None):
    
    topic = preprocess_text(topic)
    
    # Transform the topic into the feature space
    topic_vector = vectorizer.transform([topic])
    
    # Apply PCA if necessary
    if pca:
        topic_vector_reduced = pca.transform(topic_vector.toarray())
    else:
        topic_vector_reduced = topic_vector.toarray()
    
    # Predict the cluster for the topic
    topic_cluster = kmeans.predict(topic_vector_reduced)
    
    # Find trials in the same cluster
    cluster_label = topic_cluster[0]
    related_trials = data[data['cluster'] == cluster_label]
    
    return related_trials

def generate_model_based_summary(trials):
    text = ' '.join(trials['cleaned_text'].dropna())
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs, 
        max_new_tokens=150,  
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_trial_description(trial):
    """Generate a description for a single trial using the model and tokenizer."""
    # Prepare input text
    input_text = f"Title: {trial['Study Title']}\nDescription: {trial['Brief Summary']}\n"
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)
    
    # Generate description
    summary_ids = model.generate(inputs, max_length=300, num_beams=4, early_stopping=True)
    description = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return description

def generate_combined_summary(trials):
    """Generate a combined summary for all trials."""
    trial_summaries = []
    
    for _, trial in trials.iterrows():
        # Generate description for each trial
        description = generate_trial_description(trial)
        
        # Prepare trial summary
        trial_summary = (
            f"Title: {trial.get('Study Title', 'N/A')}\n"
            f"Description: {description}\n"
            f"Start Date: {trial.get('Start Date', 'N/A')}\n"
            f"Completion Status: {trial.get('Study Status', 'N/A')}\n"
            f"Outcome: {trial.get('Results First Posted', 'N/A')}\n"
        )
        
        trial_summaries.append(trial_summary)
    
    combined_summary = "\n\n".join(trial_summaries)
    return combined_summary



def extract_diseases(text):
    doc = nlp(text)
    diseases = [ent.text for ent in doc.ents if ent.label_ == "Disease"]
    return ', '.join(diseases)

def generate_audio(text):
    tts = gTTS(text, lang='en')
    audio_path = os.path.join('static', 'audio', 'summary.mp3')
    
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    tts.save(audio_path)
    return 'audio/summary.mp3'

def save_plot_image(func, *args, **kwargs):
    img_path = kwargs.pop('img_path')
    plt.figure(figsize=(10, 6))
    func(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

def plot_date_distribution(trials, date_column, title):
    dates = pd.to_datetime(trials[date_column].dropna(), errors='coerce')
    dates = dates[dates.notnull()]  # Remove invalid dates
    plt.hist(dates, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

def plot_status_distribution(trials):
    status_counts = trials['Study Status'].value_counts()
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Study Status Distribution')
    plt.axis('equal')

def plot_clusters(text_data, n_clusters=5):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(text_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)

        if min(X.shape[0], X.shape[1]) > 1:
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X.toarray())

            plt.figure(figsize=(10, 6))
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', marker='o')
            plt.title("Cluster Visualization")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.colorbar()
            plt.savefig('static/images/cluster_visualization.png')
            plt.close()
            return None 
        else:
            return "Insufficient data for PCA" 
    except Exception as e:
        return str(e) 


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    topic = request.form['topic'].lower()
    
    # Preprocess and cluster trials if not done already
    data, kmeans, vectorizer, pca = preprocess_and_cluster(data_path, n_clusters=5)
    
    # Compute similarity between the topic and the trials
    trials = compute_similarity(topic, data, vectorizer)
    
    # Filter trials based on a similarity threshold (e.g., top 10 similar trials)
    similarity_threshold = 0.1
    related_trials = trials[trials['similarity'] > similarity_threshold]
    
    if related_trials.empty:
        return render_template('dashboard.html', error=f"No trials found for topic '{topic}'")

    summary_text = generate_combined_summary(related_trials)
    disease_names = extract_diseases(summary_text)
    
    audio_path = generate_audio(summary_text)
    
    table_html = related_trials[['NCT Number', 'Study Title', 'Study Status', 'Brief Summary', 'Conditions', 
                                 'Interventions', 'Secondary Outcome Measures', 'Sponsor', 'Collaborators',
                                 'Start Date', 'Completion Date', 'Results First Posted']].to_html(
        classes='table table-striped',
        index=False
    )
    
    cluster_img_path = 'static/images/cluster_visualization.png'
    cluster_error = plot_clusters(related_trials['cleaned_text'].dropna(), n_clusters=5)
    if cluster_error:
        cluster_img_path = None
        cluster_error = "Insufficient number of samples or features for PCA"
    else:
        save_plot_image(plot_clusters, text_data=related_trials['cleaned_text'].dropna(), n_clusters=5, img_path=cluster_img_path)

    start_date_img_path = 'static/images/start_date_distribution.png'
    completion_date_img_path = 'static/images/completion_date_distribution.png'
    status_img_path = 'static/images/status_distribution.png'
    
    save_plot_image(plot_date_distribution, related_trials, 'Start Date', 'Start Date Distribution', img_path=start_date_img_path)
    save_plot_image(plot_date_distribution, related_trials, 'Completion Date', 'Completion Date Distribution', img_path=completion_date_img_path)
    save_plot_image(plot_status_distribution, related_trials, img_path=status_img_path)

    return render_template('dashboard.html', topic=topic, summary=summary_text, 
                           table_html=table_html, diseases=disease_names, 
                           cluster_img_path=cluster_img_path, cluster_error=cluster_error, 
                           audio_path=audio_path, start_date_img_path=start_date_img_path, 
                           completion_date_img_path=completion_date_img_path, 
                           status_img_path=status_img_path, trials=related_trials.to_dict(orient='records'))


@app.route('/audio/<filename>')
def audio(filename):
    return send_from_directory(os.path.join('static', 'audio'), filename)

@app.route('/images/<filename>')
def images(filename):
    return send_from_directory(os.path.join('static', 'images'), filename)

if __name__ == '__main__':
    app.run(debug=True)