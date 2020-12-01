import nltk, string
from math import sqrt
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')
nltk.download('stopwords')

punct = list(string.punctuation)
stop_words = set(stopwords.words('english'))

def get_xml():
    with open('news.xml', 'r', encoding='utf8') as inf:
        return bs(inf.read(), 'html.parser')

def splitter(doc):
    return doc.split()

def get_tfidf_indices(text, volume):
    vectorizer = TfidfVectorizer(tokenizer=splitter, norm=None)
    X = vectorizer.fit_transform(text)
    features = vectorizer.get_feature_names()
    for i, sentence in enumerate(text):
        sentence = sentence.split()
        n = len(sentence)
        for word in set(sentence):
            j = features.index(word)
            X[i, j] /= n
    X = X.sum(axis=1)
    res = X.argsort(0)[-volume:].reshape(volume).tolist()[0]
    res.sort()
    return res

xml = get_xml()

lemmatizer = WordNetLemmatizer()

for tag in xml.find_all('news'):
    header = tag.contents[1].contents[0]
    body = [line.strip() for line in tag.contents[3].contents[0].split('\n')]
    sentences = []
    for s in body:
        sentence = []
        for word in word_tokenize(s):
            lem = lemmatizer.lemmatize(word).lower()
            if lem in punct or lem in stop_words:
                continue
            sentence.append(lem)
        sentences.append(' '.join(sentence))
    volume = round(sqrt(len(body)))
    inds = get_tfidf_indices(sentences, volume)
    res = '\n'.join([body[i] for i in inds])
    print('HEADER: {}\nTEXT: {}\n'.format(header, res))
