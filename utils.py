import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from typing import List, Tuple, Dict, Any


class SemanticClassifier:
    """Semantic column classifier using embeddings and cosine similarity"""
    
    def __init__(self, training_data_path: str = "TrainingData/TrainingData"):
        self.training_data_path = training_data_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.category_prototypes = {}
        self.countries_data = []
        self.legal_suffixes = []
        self._load_training_data()
        self._build_prototypes()
    
    def _load_training_data(self):
        """Load all training datasets"""
        # Load company data
        company_path = os.path.join(self.training_data_path, "company.csv")
        self.company_data = pd.read_csv(company_path)['company'].dropna().tolist()
        
        # Load countries data
        countries_path = os.path.join(self.training_data_path, "countries.txt")
        with open(countries_path, 'r', encoding='utf-8') as f:
            self.countries_data = [line.strip().lower() for line in f.readlines() if line.strip()]
        
        # Load dates data
        dates_path = os.path.join(self.training_data_path, "dates.csv")
        dates_df = pd.read_csv(dates_path)
        self.dates_data = dates_df['date'].dropna().tolist()
        
        # Load phone data
        phone_path = os.path.join(self.training_data_path, "phone.csv")
        phone_df = pd.read_csv(phone_path)
        self.phone_data = phone_df['number'].dropna().tolist()
        
        # Load legal suffixes
        legal_path = os.path.join(self.training_data_path, "legal.txt")
        with open(legal_path, 'r', encoding='utf-8') as f:
            self.legal_suffixes = [line.strip().lower() for line in f.readlines() if line.strip()]
    
    def _build_prototypes(self):
        """Build prototype embeddings for each category"""
        print("Building category prototypes...")
        
        # Sample data for prototyping (limit to avoid memory issues)
        sample_size = 100
        
        categories = {
            'phone': self.phone_data[:sample_size],
            'company': self.company_data[:sample_size],
            'country': self.countries_data[:sample_size],
            'date': self.dates_data[:sample_size]
        }
        
        for category, data in categories.items():
            # Filter out invalid/empty values
            valid_data = [str(item) for item in data if item and str(item).strip()]
            if valid_data:
                embeddings = self.model.encode(valid_data)
                # Use mean embedding as prototype
                self.category_prototypes[category] = np.mean(embeddings, axis=0)
            else:
                # Fallback: use category name as prototype
                self.category_prototypes[category] = self.model.encode([category])[0]
        
        print(f"Built prototypes for {len(self.category_prototypes)} categories")
    
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]
    
    def classify_column(self, column_values: List[str], sample_size: int = 50) -> str:
        """Classify a column based on semantic similarity to prototypes"""
        # Clean and sample the data
        valid_values = [str(val).strip() for val in column_values if val and str(val).strip()]
        if not valid_values:
            return 'other'
        
        # Sample values to avoid computational overhead
        if len(valid_values) > sample_size:
            sample_indices = np.random.choice(len(valid_values), sample_size, replace=False)
            sampled_values = [valid_values[i] for i in sample_indices]
        else:
            sampled_values = valid_values
        
        # Generate embeddings for sampled values
        value_embeddings = self.embed_text(sampled_values)
        mean_embedding = np.mean(value_embeddings, axis=0)
        
        # Compute similarities with each category prototype
        similarities = {}
        for category, prototype in self.category_prototypes.items():
            similarity = self.compute_similarity(mean_embedding, prototype)
            similarities[category] = similarity
        
        # Find the category with highest similarity
        best_category = max(similarities, key=similarities.get)
        best_score = similarities[best_category]
        
        # If similarity is too low, classify as 'other'
        threshold = 0.3  # Adjust based on experimentation
        if best_score < threshold:
            return 'other'
        
        return best_category
    
    def parse_phone_number(self, phone_value: str) -> Tuple[str, str]:
        """Parse phone number into country and number using semantic similarity"""
        phone_value = str(phone_value).strip()
        
        # Embed the phone value
        phone_embedding = self.embed_text([phone_value])[0]
        
        # Embed all countries
        country_embeddings = self.embed_text(self.countries_data)
        
        # Find most similar country
        similarities = []
        for i, country_emb in enumerate(country_embeddings):
            sim = self.compute_similarity(phone_embedding, country_emb)
            similarities.append((sim, self.countries_data[i]))
        
        # Get the best matching country
        best_similarity, best_country = max(similarities, key=lambda x: x[0])
        
        # If similarity is too low, try to extract country from phone value semantically
        if best_similarity < 0.2:
            # Split phone value into tokens and find most country-like token
            tokens = phone_value.replace('+', ' ').replace('-', ' ').replace('(', ' ').replace(')', ' ').split()
            if tokens:
                token_similarities = []
                for token in tokens:
                    if len(token) > 1:  # Skip single characters
                        token_emb = self.embed_text([token])[0]
                        max_country_sim = max([
                            self.compute_similarity(token_emb, country_emb) 
                            for country_emb in country_embeddings
                        ])
                        token_similarities.append((max_country_sim, token))
                
                if token_similarities:
                    _, best_token = max(token_similarities, key=lambda x: x[0])
                    # Find which country this token is most similar to
                    token_emb = self.embed_text([best_token])[0]
                    country_sims = [(self.compute_similarity(token_emb, ce), country) 
                                  for ce, country in zip(country_embeddings, self.countries_data)]
                    _, best_country = max(country_sims, key=lambda x: x[0])
        
        # Extract number part (everything that's not the country)
        # Use semantic approach: remove the most country-like parts
        tokens = phone_value.replace('+', ' ').replace('-', ' ').replace('(', ' ').replace(')', ' ').split()
        number_tokens = []
        
        for token in tokens:
            if len(token) > 1:
                token_emb = self.embed_text([token])[0]
                # Check if token is more similar to numbers than countries
                is_numeric = any(char.isdigit() for char in token)
                country_sim = max([self.compute_similarity(token_emb, ce) for ce in country_embeddings])
                
                # Keep token if it's more numeric-like or less country-like
                if is_numeric or country_sim < 0.3:
                    number_tokens.append(token)
        
        number_part = ' '.join(number_tokens) if number_tokens else phone_value
        
        return best_country.title(), number_part
    
    def parse_company_name(self, company_value: str) -> Tuple[str, str]:
        """Parse company name into name and legal suffix using semantic similarity"""
        company_value = str(company_value).strip()
        
        # Embed legal suffixes
        legal_embeddings = self.embed_text(self.legal_suffixes)
        
        # Split company name into tokens
        tokens = company_value.lower().replace(',', ' ').replace('.', ' ').split()
        
        if not tokens:
            return company_value, ""
        
        # Find the token most similar to legal suffixes
        best_legal_similarity = 0
        best_legal_token_idx = -1
        best_legal_suffix = ""
        
        for i, token in enumerate(tokens):
            token_emb = self.embed_text([token])[0]
            for j, legal_emb in enumerate(legal_embeddings):
                similarity = self.compute_similarity(token_emb, legal_emb)
                if similarity > best_legal_similarity:
                    best_legal_similarity = similarity
                    best_legal_token_idx = i
                    best_legal_suffix = self.legal_suffixes[j]
        
        # If we found a good legal suffix match
        if best_legal_similarity > 0.4:  # Threshold for legal suffix
            # Name is everything before the legal suffix
            name_tokens = tokens[:best_legal_token_idx]
            name_part = ' '.join(name_tokens).strip()
            legal_part = best_legal_suffix
        else:
            # No clear legal suffix found
            name_part = company_value
            legal_part = ""
        
        return name_part.title(), legal_part.upper()


def load_column_from_file(file_path: str, column_name: str) -> List[str]:
    """Load a specific column from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        if column_name in df.columns:
            return df[column_name].tolist()
        else:
            raise ValueError(f"Column '{column_name}' not found in file. Available columns: {list(df.columns)}")
    except Exception as e:
        raise Exception(f"Error loading file: {e}")


def sample_column_values(values: List[str], sample_size: int = 100) -> List[str]:
    """Sample values from a column for classification"""
    valid_values = [v for v in values if v and str(v).strip()]
    if len(valid_values) <= sample_size:
        return valid_values
    return np.random.choice(valid_values, sample_size, replace=False).tolist()
