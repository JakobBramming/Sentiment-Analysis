from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
from ground_truth import ground_truth
from sklearn.metrics import classification_report

# Load pre-trained model and tokenizer from HuggingFace
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Zero-shot classifier using cosine distance
def zsc_cosine(sentence, labels):
    sentence_embedding = get_embeddings(sentence)
    label_embeddings = [get_embeddings(label) for label in labels]
    similarities = [cosine_similarity(sentence_embedding, label_embedding) for label_embedding in label_embeddings]
    return labels[np.argmax(similarities)]

# Zero-shot classifier using euclidean distance
def zsc_euclidean(sentence, labels):
    sentence_embedding = get_embeddings(sentence)
    label_embeddings = [get_embeddings(label) for label in labels]
    similarities = [euclidean_distances(sentence_embedding, label_embedding) for label_embedding in label_embeddings]
    return labels[np.argmin(similarities)]


if __name__ == "__main__":

    # Reading sentences from a text file
    with open('sentences.txt', 'r') as file:
        sentences = [line.strip() for line in file]

    # Reading labels from a text file
    with open('labels.txt', 'r') as file:
        labels = [line.strip() for line in file]

    # Get embeddings for each sentence
    sentence_embeddings = [get_embeddings(sentence) for sentence in sentences]

    #importing true labels for perfomance testing
    true_label = [label for _, label in ground_truth]

    # Applying zero-shot classifiers on the data
    results_cosine = []
    results_euclidean = []

    for i, sentence in enumerate(sentences):
        euclidean_label = zsc_euclidean(sentence, labels)
        #print(f"Predicted label for '{sentence}': {euclidean_label}")
        results_euclidean.append((sentence, true_label[i], euclidean_label))

    for i, sentence in enumerate(sentences):
        cosine_label = zsc_cosine(sentence, labels)
        #print(f"Predicted label for '{sentence}': {cosine_label}")
        results_cosine.append((sentence, true_label[i], cosine_label))

    # Classification report for cosine classifier
    y_true_cosine = [true_label for _, true_label, _ in results_cosine]
    y_pred_cosine = [cosine_label for _, _, cosine_label in results_cosine] 
    print(classification_report(y_true_cosine, y_pred_cosine))

    # Classification report for euclidean classifier
    y_true_euclidean = [true_label for _, true_label, _ in results_euclidean]
    y_pred_euclidean = [euclidean_label for _, _, euclidean_label in results_euclidean]
    print(classification_report(y_true_euclidean, y_pred_euclidean))

    # Example for how one sentence compares to all labels
    def show_cosine_distances(sentence, labels):
        sentence_embedding = get_embeddings(sentence)
        label_embeddings = [get_embeddings(label) for label in labels]
        distances = [cosine_similarity(sentence_embedding, label_embedding)[0][0] for label_embedding in label_embeddings]
        for label, distance in zip(labels, distances):
            print(f"Cosine distance from '{sentence}' to '{label}': {distance:.4f}")

    # Example usage
    example_sentence = sentences[92]  # Change this to any sentence you want to test
    show_cosine_distances(example_sentence, labels)
    

    # Plot the cosine similarities between mean word embeddings and sentence embeddings
    # to determine if the model is capturing the semantics of the sentences

    # Calculate mean of word embeddings for each sentence
    mean_word_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        mean_word_embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

    # Compare mean of word embeddings with sentence embeddings
    similarities = []
    for i, sentence in enumerate(sentences):
        similarity = cosine_similarity(mean_word_embeddings[i], sentence_embeddings[i])
        similarities.append(similarity[0][0])
        print(f"Sentence: {sentence}")
        print(f"Cosine similarity: {similarity[0][0]}")

    # Plot the similarities
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(range(len(sentences))), y=similarities)
    plt.xlabel('Sentence Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between Mean Word Embeddings and Sentence Embeddings')
    plt.show()