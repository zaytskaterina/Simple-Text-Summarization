import nltk, string
from math import sqrt
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')
punct = list(string.punctuation)
stop_words = set(stopwords.words('english'))

def get_xml():
    with open('news.xml', 'r', encoding='utf8') as inf:
        return bs(inf.read(), 'html.parser')

def choose_sentence(text, d):
    weights = []
    h_word = max(d, key=lambda k: d[k])
    for sent in text:
        w = 0
        n = len(sent)
        for word in sent:
            w += d[word]
        weights.append(w / n)
    indexes = [weights.index(ind) for ind in sorted(weights, reverse=True)]
    for ind in indexes:
        if h_word in text[ind]:
            i = ind
            break
    for word in text[i]:
        d[word] = d[word] ** 2
    return (i, d)

xml = get_xml()

lemmatizer = WordNetLemmatizer()

for tag in xml.find_all('news'):
    d = {}
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
            if lem not in d.keys():
                d[lem] = 1
            else:
                d[lem] += 1
        sentences.append(sentence)
    N = sum(d.values())
    for key, value in d.items():
        d[key] = value / N
    volume = round(sqrt(len(body)))
    inds = []
    for i in range(volume):
        n, d = choose_sentence(sentences, d)
        inds.append(n)
    res = '\n'.join([body[i] for i in sorted(inds)])
    print('HEADER: {}\nTEXT: {}\n'.format(header, res))
