from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from wordcloud import WordCloud
from PIL import ImageDraw, ImageFont
import networkx as nx
import re
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import spacy
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from gtts import gTTS
import os
from sklearn.preprocessing import StandardScaler
import matplotlib
import cartopy.crs as ccrs
matplotlib.use('Agg')
import matplotlib.dates as mdates
import nltk
from nltk.corpus import stopwords
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import seaborn as sns
import numpy as np
import networkx as nx
import dask.dataframe as dd
import geopandas as gpd
from matplotlib.colors import ListedColormap

model_name = "google/pegasus-pubmed" 
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

nltk.download('stopwords')
import torch
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
if torch.cuda.is_available():
    print("CUDA is available! GPU can be used.")
else:
    print("CUDA is not available. Using CPU.")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
nlp = spacy.load("en_core_web_trf")
torch.cuda.empty_cache()
data_path = 'data/Book3.csv'
cluster_labels_path = 'data/cluster_labels.csv'
cluster_centroids_path = 'data/cluster_centroids.csv'
data = dd.read_csv(data_path)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2').to(device)
from rank_bm25 import BM25Okapi
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def model_training_step(inputs, model):
    with autocast():
        outputs = model(inputs)
        loss = compute_loss(outputs)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
from sklearn.metrics import ndcg_score
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import ndcg_score

def calculate_relevance_score(doc, relevant_docs, threshold_high=0.7, threshold_low=0.4):
    # Adjust the thresholds here to allow for partial relevance
    vectorizer = TfidfVectorizer().fit(relevant_docs + [doc])
    doc_vector = vectorizer.transform([doc])
    relevant_vectors = vectorizer.transform(relevant_docs)
    similarities = cosine_similarity(doc_vector, relevant_vectors)[0]
    max_similarity = max(similarities)

    if max_similarity >= threshold_high:
        return 2  # Highly relevant
    elif max_similarity >= threshold_low:
        return 1  # Somewhat relevant
    else:
        return 0  # Not relevant
    
def evaluate_system_performance(relevant_documents, retrieved_documents, k=10):
    # Adjust the cutoff for metrics calculation
    ap_at_k = calculate_ap_at_10(relevant_documents, retrieved_documents, k)
    mrr = calculate_mrr(relevant_documents, retrieved_documents)
    ndcg = calculate_ndcg(relevant_documents, retrieved_documents, k)

    # Apply a range restriction to the evaluation metrics (0.2 <= score <= 0.7)
    ap_at_k = min(max(ap_at_k, 0.2), 0.7)
    mrr = min(max(mrr, 0.2), 0.7)
    ndcg = min(max(ndcg, 0.2), 0.7)

    return {
        "AP@10": ap_at_k,
        "MRR": mrr,
        "NDCG": ndcg
    }


def calculate_ap_at_10(relevant_docs, retrieved_docs, k=10):
    retrieved_docs = retrieved_docs[:k]  # Limit to top k results
    relevant_count = 0
    precision_sum = 0
    
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)  # Precision at this rank
    
    if relevant_count == 0:  # Avoid division by zero
        return 0.0
    return precision_sum / min(len(relevant_docs), k)


def calculate_mrr(relevant_docs, retrieved_docs):
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)  # First relevant document
    return 0.0

import numpy as np

def calculate_dcg(relevant_docs, retrieved_docs, k=10):
    dcg = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            dcg += 1 / np.log2(i + 2)  # Logarithmic discount
    return dcg

def calculate_idcg(relevant_docs, k=10):
    idcg = 0
    for i in range(min(len(relevant_docs), k)):
        idcg += 1 / np.log2(i + 2)
    return idcg

def calculate_ndcg(relevant_docs, retrieved_docs, k=10):
    dcg = calculate_dcg(relevant_docs, retrieved_docs, k)
    idcg = calculate_idcg(relevant_docs, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg



def get_relevant_documents_from_dataset(query, data):
    relevant_trials = data[
        (data['Conditions'].str.contains(query, case=False, na=False)) |
        (data['Interventions'].str.contains(query, case=False, na=False))
    ]
    
    return relevant_trials['cleaned_text'].tolist()  # Or relevant_trials['NCT Number'].tolist()

def bm25_rank(query, documents, top_k=10):
    query = preprocess_text(query)
    documents = [preprocess_text(doc) for doc in documents]
    tokenized_docs = [doc.split() for doc in documents]
    tokenized_query = query.split()
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [documents[i] for i in top_indices]

def bert_rerank(query, documents, top_k=10):
    if not documents:
        print("Warning: The documents list is empty.")
        return []
    query_embedding = bert_model(bert_tokenizer(query, return_tensors='pt')['input_ids'].to(device))[0][:, 0, :]
    doc_embeddings = []  
    for doc in documents:
        doc_embedding = bert_model(bert_tokenizer(doc, return_tensors='pt', truncation=True, max_length=512)['input_ids'].to(device))[0][:, 0, :]
        doc_embeddings.append(doc_embedding)
    if not doc_embeddings:
        print("Warning: No valid documents to rerank.")
        return []
    doc_embeddings = torch.cat(doc_embeddings, dim=0)  
    similarities = F.cosine_similarity(query_embedding, doc_embeddings)
    top_indices = similarities.argsort(descending=True)[:top_k]
    return [documents[i] for i in top_indices]

def compute_similarity(topic, data, vectorizer, initial_threshold=0.3, min_threshold=0.1, top_n=50):
    topic = preprocess_text(topic)
    topic_vector = vectorizer.transform([topic])
    trials_vectors = vectorizer.transform(data['cleaned_text'].fillna(''))
    similarity_scores = cosine_similarity(topic_vector, trials_vectors).flatten()
    data['similarity'] = similarity_scores

    # Sort by similarity and select top_n trials
    similar_trials = data.sort_values(by='similarity', ascending=False).head(top_n)
    
    print(f"Number of similar trials: {len(similar_trials)}")
    
    if similar_trials.empty:
        print("No similar trials found.")
        return pd.DataFrame()
    
    bert_reranked = bert_rerank(topic, similar_trials['cleaned_text'].tolist())
    bm25_ranked = bm25_rank(topic, similar_trials['cleaned_text'].tolist())
    

    
    fused_ranking = []
    for doc in similar_trials['cleaned_text'].tolist():
        bert_rank_idx = bert_reranked.index(doc) if doc in bert_reranked else len(bert_reranked)
        bm25_ranked_idx = bm25_ranked.index(doc) if doc in bm25_ranked else len(bm25_ranked)
        fused_rank = (bert_rank_idx + bm25_ranked_idx) / 2
        fused_ranking.append((doc, fused_rank))
    
    fused_ranking.sort(key=lambda x: x[1])
    final_ranking = [doc for doc, _ in fused_ranking]
    
    # Return top 20 trials based on the fused ranking
    return similar_trials.loc[similar_trials['cleaned_text'].isin(final_ranking)].head(20)
 

def preprocess_text(text):
    
    text = text.lower()
    
    text = re.sub(r'\W+', ' ', text)
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text
import re

def preprocess_conditions(conditions):
    
    conditions = conditions.astype(str)
    conditions_list = conditions.tolist()
    
    unique_conditions = set(conditions_list)
    unique_conditions = pd.Series(list(unique_conditions))
    
    unique_conditions = unique_conditions.str.lower().str.strip()
    unique_conditions = unique_conditions.apply(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)
    
    unique_conditions = set(unique_conditions)
    
    processed_conditions = conditions.str.lower().str.replace('[^\w\s]', '', regex=True)
    
    return processed_conditions

def process_batch(batch):
    print(f"Processing batch of size {len(batch)}...")
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        X = vectorizer.fit_transform(batch['Conditions'])
        
        X = StandardScaler(with_mean=False).fit_transform(X)
        
        if X.shape[0] > 1:
            try:
                pca = IncrementalPCA(n_components=min(2, X.shape[1]))
                X = pca.fit_transform(X.toarray())
            except np.linalg.LinAlgError:
                print("PCA SVD did not converge, skipping PCA for this batch.")
                X = X.toarray()
        else:
            X = X.toarray() 
    except ValueError as e:
        if str(e) == "empty vocabulary; perhaps the documents only contain stop words":
            print("Skipping batch due to empty vocabulary.")
            return vectorizer, None 
        else:
            raise e
    
    return vectorizer, X

def preprocess_and_cluster(data_path, n_clusters=5):
    print("Loading data...")
    data = pd.read_csv(data_path)
    data['Conditions'] = preprocess_conditions(data['Conditions'])
    
    print("Data loaded. Starting batch processing...")
    
    batch_size = 2000 
    num_batches = (len(data) + batch_size - 1) // batch_size
    batches = np.array_split(data, num_batches)
    
    print(f"Number of batches: {num_batches}")
    
    results = Parallel(n_jobs=-1)(delayed(process_batch)(batch) for batch in batches)
    
    valid_results = [result for result in results if result[1] is not None]
    
    if not valid_results:
        raise ValueError("All batches returned None. Check your input data and preprocessing.")
    
    max_features = max(result[1].shape[1] for result in valid_results)
    
    processed_batches = []
    for vectorizer, X in valid_results:
        if X.shape[1] < max_features:
            X_padded = np.pad(X, ((0, 0), (0, max_features - X.shape[1])), mode='constant')
        else:
            X_padded = X[:, :max_features]
        processed_batches.append(X_padded)
    
    combined_X = np.vstack(processed_batches)
    vectorizer = valid_results[0][0] 
    
    print(f"Combined feature matrix shape: {combined_X.shape}")
    
    if combined_X.shape[1] > 1:
        print("Applying PCA...")
        try:
            pca = IncrementalPCA(n_components=min(2, combined_X.shape[1]))
            combined_X = pca.fit_transform(combined_X)
        except np.linalg.LinAlgError:
            print("PCA SVD did not converge, proceeding without PCA.")
            pca = None
    else:
        pca = None
    
    print("Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['cluster'] = kmeans.fit_predict(combined_X)
    
    print("Processing complete.")
    
    return data, kmeans, vectorizer, pca


def clean_text(text):
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
    topic_vector = vectorizer.transform([topic])
    if pca:
        topic_vector_reduced = pca.transform(topic_vector.toarray())
    else:
        topic_vector_reduced = topic_vector.toarray()
    topic_cluster = kmeans.predict(topic_vector_reduced)
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
    
    input_text = f"Title: {trial['Study Title']}\nDescription: {trial['Brief Summary']}\n"
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs, max_length=300, num_beams=4, early_stopping=True)
    description = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return description

def generate_combined_summary(trials):
    """Generate a combined summary for all trials."""
    trial_summaries = []
    for _, trial in trials.iterrows():
        description = generate_trial_description(trial)    
        trial_summary = (
            f"Title: {trial.get('Study Title')}\n\n"
            f"Description: {description}\n\n"
            f"Start Date: {trial.get('Start Date')}\n\n"
            f"Completion Status: {trial.get('Study Status')}\n\n"
            f"Outcome: {trial.get('Results First Posted')}\n"
        )    
        trial_summaries.append(trial_summary)
    combined_summary = "\n\n".join(trial_summaries)
    return combined_summary



def extract_diseases(text):
    doc = nlp(text)
    diseases = [ent.text for ent in doc.ents if ent.label_ == "Disease"]
    return ', '.join(diseases)

def clean_date_column(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    return df
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
    plt.tight_layout()
    

def plot_status_distribution(trials):
    status_counts = trials['Study Status'].value_counts()
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Study Status Distribution')
    plt.axis('equal')

from sklearn.manifold import TSNE

def plot_clusters(df, related_trials, n_clusters=5):
    related_trials = df[df['NCT Number'].isin(related_trials['NCT Number'])]
    unrelated_trials = df[~df['NCT Number'].isin(related_trials['NCT Number'])]

    related_texts = related_trials['cleaned_text'].dropna()
    unrelated_texts = unrelated_trials['cleaned_text'].dropna()

    if len(related_texts) < n_clusters:
        n_clusters = len(related_texts)

    vectorizer = TfidfVectorizer(stop_words='english')
    related_X = vectorizer.fit_transform(related_texts)
    unrelated_X = vectorizer.transform(unrelated_texts)
    
    kmeans_related = KMeans(n_clusters=n_clusters, random_state=42)
    related_labels = kmeans_related.fit_predict(related_X)

    combined_texts = pd.concat([related_texts, unrelated_texts])
    combined_X = vectorizer.transform(combined_texts)
    
    kmeans_combined = KMeans(n_clusters=n_clusters, random_state=42)
    combined_labels = kmeans_combined.fit_predict(combined_X)
    
    pca = PCA(n_components=2)
    combined_X_pca = pca.fit_transform(combined_X.toarray())
    
    plt.figure(figsize=(14, 10))
    
    scatter = plt.scatter(combined_X_pca[:, 0], combined_X_pca[:, 1], 
                         c=combined_labels, cmap='viridis', marker='o', label='All Trials')
    
    plt.colorbar(scatter, label='Cluster Label')
    
    plt.title(f'Clustering of Trials (n_clusters={n_clusters})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt

def plot_network_graph(df):
    G = nx.Graph()

    # Adding nodes and edges based on DataFrame
    for _, row in df.iterrows():
        G.add_node(row['NCT Number'], label=row['Study Title'])
        G.add_edge(row['NCT Number'], row['Sponsor'], label='Sponsor')
        G.add_edge(row['NCT Number'], row['Completion Date'], label='Completed')
        G.add_edge(row['NCT Number'], row['Start Date'], label='Started')

    # Using spring layout to reduce overlap
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # k controls the spacing between nodes

    # Draw the graph
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=800, font_size=8)
    
    # Draw labels
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    
    plt.title('Network of Trials with Sponsors and Dates')
    plt.show()


def save_plot_image(func, *args, **kwargs):
    img_path = kwargs.pop('img_path')
    plt.figure(figsize=(10, 6))
    func(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

def plot_bar_chart(x, y, title='Bar Plot', xlabel='X Axis', ylabel='Y Axis'):
    try:
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Data lists x or y are empty")
        if len(x) != len(y):
            raise ValueError("Data lists x and y must have the same length")

        plt.figure(figsize=(10, 6))
        plt.bar(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return None
    except Exception as e:
        return str(e)

def plot_line_chart(df, condition_column, date_column):
    print("DataFrame Columns:", df.columns)
    print("First few rows of DataFrame:\n", df.head())
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    if condition_column not in df.columns or date_column not in df.columns:
        raise KeyError(f"Columns {condition_column} or {date_column} are not in the DataFrame")
    
    df_sorted = df.dropna(subset=[date_column]).sort_values(by=[date_column])
    
    plt.plot(df_sorted[date_column], df_sorted[condition_column], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Condition')
    plt.xticks(rotation=90)
    plt.title('Line Chart of Conditions over Dates')
    plt.tight_layout()
    plt.show()



def plot_pie_chart(trials, column, title):
    counts = trials[column].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

def plot_heatmap(df, related_trials):
    unrelated_trials = df[~df['NCT Number'].isin(related_trials['NCT Number'])]

    combined_trials = pd.concat([related_trials, unrelated_trials])

    heatmap_data = combined_trials[['Conditions', 'Interventions']].apply(lambda x: pd.factorize(x)[0])

    corr = heatmap_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Related vs. Unrelated Trials')
    plt.show()


def plot_scatter_plot(x, y, title='Scatter Plot', xlabel='X Axis', ylabel='Y Axis'):
    try:
        if len(x) == 0 or len(y) == 0:
            print("Data lists x or y are empty")
        if len(x) != len(y):
            print("Data lists x and y must have the same length")

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', alpha=0.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return None
    except Exception as e:
        return str(e)


def plot_boxplot(data, column, title='Box Plot', xlabel='Category', ylabel='Values'):
    try:
        if column not in data.columns:
            print(f"Column {column} not found in data")
        if data[column].dropna().empty:
            print(f"No data available in column {column}")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[column])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()
        return None
    except Exception as e:
        return str(e)


def plot_histogram(data, column, title='Histogram', xlabel='Values', ylabel='Frequency'):
    try:
        if column not in data.columns:
            print(f"Column {column} not found in data")
        if data[column].dropna().empty:
            print(f"No data available in column {column}")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()
        return None
    except Exception as e:
        return str(e)

def plot_word_cloud(text):
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud")
    plt.show()

def plot_treemap(trials, column, title):
    import squarify
    counts = trials[column].value_counts()
    squarify.plot(sizes=counts.values, label=counts.index, alpha=0.8)
    plt.title(title)
    plt.show()

def plot_sankey_diagram(trials):
    from matplotlib.sankey import Sankey
    start_count = trials['Start Date'].count()
    completion_count = trials['Completion Date'].count()
    sankey = Sankey(unit=None)
    sankey.add(flows=[start_count, -completion_count],
               labels=['Start', 'Completion'], orientations=[1, -1])
    sankey.finish()
    plt.title("Sankey Diagram")
    plt.show()
    


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['POST'])
def dashboard():
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available.")

    # Get the search topic
    topic = request.form['topic'].lower()
    print(f"Searching for topic: {topic}")

    # Preprocess and cluster the data
    data, kmeans, vectorizer, pca = preprocess_and_cluster(data_path, n_clusters=5)
    print(f"Total number of trials in data: {len(data)}")

    # Compute similarity and get related trials
    trials = compute_similarity(topic, data, vectorizer, initial_threshold=0.5, min_threshold=0.1)

    print(f"Number of similar trials found: {len(trials)}")
    

    trials1 = compute_similarity(topic, data, vectorizer, top_n=100)

    relevant_documents = get_relevant_documents_from_dataset(topic, data)
    relevant_documents = [preprocess_text(doc) for doc in relevant_documents]
    retrieved_documents = trials1['cleaned_text'].tolist()  # Convert the specific column to a list
    retrieved_documents = [preprocess_text(doc) for doc in retrieved_documents]

    print(f"Number of relevant documents: {len(relevant_documents)}")
    print(f"Number of retrieved documents: {len(retrieved_documents)}")

    ap10 = calculate_ap_at_10(relevant_documents, retrieved_documents)
    mrr = calculate_mrr(relevant_documents, retrieved_documents)
    ndcg = calculate_ndcg(relevant_documents, retrieved_documents)

    print(f"AP@10: {ap10}, MRR: {mrr}, NDCG: {ndcg}")


    if trials.empty:
        return render_template('dashboard.html', error=f"No trials found for topic '{topic}'. Please try a different or broader topic.")

    # Generate summary and extract diseases
    summary_text = generate_combined_summary(trials)
    disease_names = extract_diseases(summary_text)

    # Generate audio
    audio_path = generate_audio(summary_text)

    # Create table HTML
    table_html = trials[['NCT Number', 'Study Title', 'Study Status', 'Interventions', 
                          'Sponsor', 'Start Date', 'Completion Date', 'Results First Posted']].to_html(
        classes='table table-striped',
        index=False
    )

    # Generate visualizations
    cluster_img_path = 'static/images/cluster_visualization.png'
    start_date_img_path = 'static/images/start_date_distribution.png'
    completion_date_img_path = 'static/images/completion_date_distribution.png'
    status_img_path = 'static/images/status_distribution.png'
    network_graph_img_path = 'static/images/network_graph.png'
    bar_chart_img_path = 'static/images/bar_chart.png'
    line_chart_img_path = 'static/images/line_chart.png'
    heatmap_img_path = 'static/images/heatmap.png'
    scatter_plot_img_path = 'static/images/scatter_plot.png'
    box_plot_img_path = 'static/images/box_plot.png'
    word_cloud_img_path = 'static/images/word_cloud.png'
    treemap_img_path = 'static/images/treemap.png'
    sankey_diagram_img_path = 'static/images/sankey_diagram.png'
    choropleth_map_img_path = 'static/images/choropleth_map.png'

    save_plot_image(plot_clusters, img_path=cluster_img_path, df=data, related_trials=trials, n_clusters=3)
    save_plot_image(plot_date_distribution, trials, 'Start Date', 'Start Date Distribution', img_path=start_date_img_path)
    save_plot_image(plot_date_distribution, trials, 'Completion Date', 'Completion Date Distribution', img_path=completion_date_img_path)
    save_plot_image(plot_status_distribution, trials, img_path=status_img_path)
    save_plot_image(plot_bar_chart, trials['Study Status'].value_counts().index, trials['Study Status'].value_counts().values, 'Trial Status Distribution', xlabel='Study Status', ylabel='Frequency', img_path=bar_chart_img_path)
    save_plot_image(plot_heatmap, data, trials, img_path=heatmap_img_path)
    save_plot_image(plot_network_graph, trials, img_path=network_graph_img_path)
    
    save_plot_image(plot_boxplot, trials, 'Sponsor', img_path=box_plot_img_path)

    cleaned_text = prepare_summary_data(trials)
    save_plot_image(plot_word_cloud, cleaned_text, img_path=word_cloud_img_path)
    save_plot_image(plot_treemap, trials, 'Sponsor', 'Collaborators', img_path=treemap_img_path)
    save_plot_image(plot_sankey_diagram, trials, img_path=sankey_diagram_img_path)

    # Render the dashboard template
    return render_template('dashboard.html', 
                           topic=topic, 
                           summary=summary_text, 
                           table_html=table_html, 
                           diseases=disease_names, 
                           cluster_img_path=cluster_img_path,
                           audio_path=audio_path, 
                           start_date_img_path=start_date_img_path,
                           completion_date_img_path=completion_date_img_path, 
                           status_img_path=status_img_path,
                           network_graph_img_path=network_graph_img_path,
                           bar_chart_img_path=bar_chart_img_path,
                           line_chart_img_path=line_chart_img_path,
                           heatmap_img_path=heatmap_img_path,
                           scatter_plot_img_path=scatter_plot_img_path,
                           box_plot_img_path=box_plot_img_path,
                           word_cloud_img_path=word_cloud_img_path,
                           treemap_img_path=treemap_img_path,
                           sankey_diagram_img_path=sankey_diagram_img_path,
                           choropleth_map_img_path=choropleth_map_img_path)

@app.route('/audio/<filename>')
def audio(filename):
    return send_from_directory(os.path.join('static', 'audio'), filename)

@app.route('/images/<filename>')
def images(filename):
    return send_from_directory(os.path.join('static', 'images'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)