import pandas as pd

from collections import Counter
from collections import OrderedDict
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import itertools as tls
import matplotlib.colors as mcolors

nltk.download('stopwords')
nltk.download('punkt')

import clean
import ldadavis
import classification

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

archivo = 'C:/ruta/'

in_date = dt.date(2020,5,25)
out_date = dt.date(2020,6,2)

limit = 500000
lang = 'es'
query = ['blackout']

#busquedas inicio

extract = query_tweets(query[0], begindate = in_date, enddate = out_date,
                       limit = limit, lang = lang)

df = pd.DataFrame(t.__dict__ for t in extract)

# finalizado

df['timestamp'] = pd.to_datetime(df['date'])
df.sort_values(by=['timestamp'], inplace=True)

df['count_tweets'] = 1
df['day'] = df['timestamp'].apply(lambda x: x.day)

df_tweets = df[['text']]
df_tweets = df_tweets.reset_index(drop=True)

strings = df_tweets.text

formated_array_data = clean.oracionToStrArr(clean.formatear(strings, pd), tls)

data_limpia = clean.formatear(strings, pd)
data_limpia = pd.concat([data_limpia, df_tweets.text], axis=1)

data_limpia.to_csv(archivo + 'dlean.csv', sep='\t')

dlk = pd.read_csv(archivo + 'dlean.csv', sep='\t', dtype=str)
dlk.rename(columns={'0': 'tweet'}, inplace=True)
df_tweets.text = dlk.tweet
dlk = dlk.tweet

dlk.to_csv(archivo + 'dlean.csv', sep='\t')

with open (archivo + 'dlean.csv','r') as miarchivo:
	texto = miarchivo.read()

stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(texto)

word_tokens = list(filter(lambda token: token not in string.punctuation, word_tokens))

myFilter = []

for word in word_tokens:
	if word not in stop_words:
		myFilter.append(word)

c = Counter(myFilter)

y = OrderedDict(c.most_common())

count_words = pd.DataFrame(c.most_common(), columns = ['word', 'count'])
count_words = count_words[count_words['count'] != 1]
count_words.to_csv(archivo + 'df_count.csv', sep=',')

topics = []
thisdict = {}

df['categoria'] = ''

classification.topics_model('text', 'categoria', df, topics, thisdict)

classification.dataframe_counts(df, 'categoria')

#mantenimiento

i = 0
while i < len(df):
	intent = df['categoria'][i]

	for topic in topics:
		if topic in intent:
			thisdict[topic][0] += int(df['retweets'][i])
			thisdict[topic][1] += int(df['replies'][i])
	i += 1

i = 0
while i < len(thisdict):
	i += 1

with open(archivo + 'rt_words.csv','w') as g:
	g.write('Topico, RT, RP\n')
	i = 0
	while i < len(thisdict):
		g.write(f'{topics[i]}, {thisdict[topics[i]][0]}, {thisdict[topics[i]][1]}\n')
		i += 1
        
words_rt = pd.read_csv(archivo + 'rt_words.csv')

#cierre de mantenimiento

corpus = []
a = []

for i in range(len(df_tweets['text'])):
        a = df_tweets['text'][i]
        corpus.append(a)

list1 = ['rt', 'RT', 'blackout', 'blackouttuesday', 'blackoutday', 'blacklivesmatter', 'blackoutuesday', 'blm', 'blacklivesmattter']

davis = ldadavis.pydavis(archivo, corpus, 9, list1)

lda = davis[0]
df_lda = davis[1]
panel = davis[2]

pyLDAvis.save_html(panel, archivo + 'listening_blackout.html')

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

ldadavis.plotlda(lda, cols, stop_words, 3, 3, 'white', archivo)

tfidf = TfidfVectorizer(min_df = 5, max_df = 0.95, max_features = 1500, stop_words = list1 + stopwords.words('english'))
tfidf.fit(df_tweets.text)
text = tfidf.transform(df_tweets.text)
    
classification.find_optimal_clusters(text, 45)

clusters = MiniBatchKMeans(n_clusters = 10, init_size = 1024, batch_size = 2048, random_state = 20).fit_predict(text)
    
classification.plot_tsne_pca(text, clusters)
            
classification.get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)
