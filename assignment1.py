import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import gensim.downloader as api

nltk.download('brown')

def train_brown_model():
    print("Preparing Brown corpus data...")
    sentences = brown.sents()
    
    print("Training Word2Vec model on Brown corpus...")
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)
    return model

def find_similar_words(model, words):
    results = {}
    for word in words:
        try:
            similar = model.wv.most_similar(word, topn=10)
            results[word] = similar
        except KeyError:
            results[word] = f"'{word}' not in vocabulary"
    return results

def find_similar_in_pretrained(words):
    print("Loading Google's pre-trained Word2Vec model...")
    google_model = api.load('word2vec-google-news-300')
    
    results = {}
    for word in words:
        try:
            similar = google_model.most_similar(word, topn=10)
            results[word] = similar
        except KeyError:
            results[word] = f"'{word}' not in vocabulary"
    return results

def main():
    target_words = ['rebellion', 'slave']
    
    print('Loading Brown Model')
    brown_model = train_brown_model()
    print('Loaded Brown Model.')
    print('Finding similar words for targets')
    brown_results = find_similar_words(brown_model, target_words)
    
    print("\nResults from Brown corpus model:")
    for word, similar_words in brown_results.items():
        print(f"\nTop 10 similar words to '{word}':")
        if isinstance(similar_words, list):
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity:.4f}")
        else:
            print(similar_words)
    
    google_results = find_similar_in_pretrained(target_words)
    
    print("\nResults from Google's pre-trained model:")
    for word, similar_words in google_results.items():
        print(f"\nTop 10 similar words to '{word}':")
        if isinstance(similar_words, list):
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity:.4f}")
        else:
            print(similar_words)

if __name__ == "__main__":
    main()
