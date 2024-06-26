import pandas as pd
import nltk
import pickle

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('customer_support_tickets_clean.csv')

# Dividir el dataset
test = df[2000:2111]
df = df[:1000]

# Función para extraer tuplas del DataFrame
def extract_tuples(df):
    tuples = []
    for index, row in df.iterrows():
        ticket_type = row['Ticket Type']
        Ticket_Description = row['Processed_Description']
        ticket_tuple = (Ticket_Description, ticket_type)
        tuples.append(ticket_tuple)
    return tuples

ticket_tuples = extract_tuples(df)
test_tuples = extract_tuples(test)

# Función para crear una bolsa de palabras
def to_bow(tickets, label='all'):
    bow = {}
    for text in tickets:
        if text[1] == label or label == 'all':
            for word in text[0].split(' '):
                if word in bow:
                    bow[word] += 1
                else:
                    bow[word] = 1
    return bow

# Crear bolsas de palabras para cada tipo de ticket
bow = to_bow(ticket_tuples)
bow_Technical_issue = to_bow(ticket_tuples, 'Technical issue')
bow_Refund_request = to_bow(ticket_tuples, 'Refund request')
bow_Cancellation_request = to_bow(ticket_tuples, 'Cancellation request')
bow_Product_inquiry = to_bow(ticket_tuples, 'Product inquiry')
bow_Billing_inquiry = to_bow(ticket_tuples, 'Billing inquiry')

# Filtrar palabras infrecuentes
new_tt = []
for text, label in ticket_tuples:
    filtered_words = [word for word in text.split() if bow.get(word, 0) <= 100]
    new_tt.append((filtered_words, label))

# Función para obtener palabras únicas
def get_unique_words(sentences):
    all_words = []
    for (description, ticket_type) in sentences:
        for word in description:
            all_words.append(word)
    return list(set(all_words))

# Función para extraer características
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in unique_words:
        features['contains(%s)' % word] = (word in document_words)
    return features

# Obtener palabras únicas y crear conjunto de entrenamiento
unique_words = get_unique_words(new_tt)
training_set = nltk.classify.apply_features(extract_features, new_tt)

# Entrenar el clasificador Naive Bayes
classifier = NaiveBayesClassifier.train(training_set)

# Guardar el clasificador y las variables necesarias
with open('bayes_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('unique_words.pkl', 'wb') as f:
    pickle.dump(unique_words, f)

print("Modelo Bayes entrenado y guardado exitosamente.")
