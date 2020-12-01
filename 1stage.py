from lxml import etree
from nltk.tokenize import sent_tokenize

def cut_n_input(string):
    tok = sent_tokenize(string.text)
    count = len(tok)
    count = round(count**0.5)
    return tok[:count]

xml_file = open("news.xml")

root = etree.parse(xml_file).getroot()

for news in root[0]:
    print('HEADER: ' +news[0].text)
    print('TEXT:',end=' ')
    print(*cut_n_input(news[1]), sep = "\n")
    #print('TEXT: '+ cut_n_input(news[1].text))


'''
states = root[0][9]
print(states[0].text)
print()
print(states[1].text)'''
