import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    file=open(file_name,"r")
    filedata= file.readlines()
    article= filedata[0].split(". ")
    sentences=[]
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
    sentences.pop()
    return sentences


def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords =[]
    sent1= [w.lower() for w in sent1]
    sent2= [w.lower() for w in sent2]
    all_words= list(set(sent1+sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1-cosine_distance(vector1,vector2)


def gen_sim_matrix(sentences,stop_words):
    similarity_matrix=np.zeros((len(sentences),len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1==idx2:
                continue
            similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
    return similarity_matrix


def generate_summary(file_name,top_n=5):
    stop_words=stopwords.words('english')
    summarize_text=[]
    sentences =  read_article(file_name)
    sentence_similarity_matrix=d=gen_sim_matrix(sentences,stop_words)
    sentence_similarity_graph=nx.from_numpy_array(sentence_similarity_matrix)
    scores =nx.pagerank(sentence_similarity_graph)
    ranked_sentence=sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
    for i in range(top_n):
        try:
            summarize_text.append(" ".join(ranked_sentence[i][1]))
        except:
            continue
    print("Summary \n"," ".join(summarize_text)+'.')

FN="kk.txt"
FILE=open(FN,"r")
print("INPUT TEXT\n",FILE.read())
generate_summary(FN,2)
