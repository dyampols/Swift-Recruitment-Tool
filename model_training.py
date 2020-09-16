import re
import pandas as pd
import numpy as np
import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

binary_dict_1 = {"I": 0, "E": 1}
binary_dict_2 = {"N": 0, "S": 1}
binary_dict_3 = {"T": 0, "F": 1}
binary_dict_4 = {"J": 0, "P": 1}

def main():
    #create df
    df = pd.read_csv('mbti_1.csv')
    df = create_df_for_insight(df)
    data = df
    list_posts = pre_process_data(data, remove_stop_words=True)
    #dump
    np.savetxt('list_posts.csv', list_posts, fmt='%s')
    #vectrorize
    vectorized_posts = vectorize_posts(list_posts)
    #append vectorized text results to df
    for i in range(len(vectorized_posts[0])):
        df['word'+str(i)] = vectorized_posts[:,i]

    binary_vals = ['I/E','N/S','T/F','J/P']
    for binary in binary_vals:
        XX = df.drop(['type','posts','I/E','N/S','T/F','J/P'], axis=1).values
        yy = df[binary].values
  
        XX_train,XX_test,yy_train,yy_test=train_test_split(XX,yy,test_size = 0.1, random_state=5)
        
        logregg = LogisticRegression()
        logregg.fit(XX_train, yy_train)
        acc_logg = round(logregg.score(XX_train, yy_train) * 100, 2)
        print(round(acc_logg,2,), "%")
        #dump model
        filename = 'finalized_model'+str(binary_vals.index(binary))+'.sav'
        pickle.dump(logregg, open(filename, 'wb'))

def create_df_for_insight(df):
    #set data frame columns for insight 
    df['word_count_per_post'] = df['posts'].apply(lambda x: len(x.split())/50)
    df['question_marks_per_post'] = df['posts'].apply(lambda x: x.count('?')/50)
    df['ellipsis_per_post'] = df['posts'].apply(lambda x: x.count('...')/50)
    df['exclmation_marks_per_post'] = df['posts'].apply(lambda x: x.count('!')/50)
    df['tagged_music_per_post'] = df['posts'].apply(lambda x: x.count('music')/50)
    df['http_per_post'] = df['posts'].apply(lambda x: x.count('http')/50)
    df['img_per_comment'] = df['posts'].apply(lambda x: (x.count('jpg')+x.count('image'))/50)
    df['emojis_per_comment'] = df['posts'].apply(lambda x: (x.count(':)') + x.count(':(') + x.count(':O') + x.count(';)') + x.count(':P') + x.count(':''(') + x.count(':-D')/50))

    #set binary values for personality types for insight 

    df['I/E'] = df['type'].astype(str).str[0]
    df['I/E'] = df['I/E'].map(binary_dict_1)
    df['N/S'] = df['type'].astype(str).str[1]
    df['N/S'] = df['N/S'].map(binary_dict_2)
    df['T/F'] = df['type'].astype(str).str[2]
    df['T/F'] = df['T/F'].map(binary_dict_3)
    df['J/P'] = df['type'].astype(str).str[3]
    df['J/P'] = df['J/P'].map(binary_dict_4)

    return df

def vectorize_posts(list_posts):
    #vectorize posts which have been processed 
    cntizer = CountVectorizer(analyzer="word", 
                                max_features=1500, 
                                tokenizer=None,    
                                preprocessor=None, 
                                stop_words=None,  
                                max_df=0.7,
                                min_df=0.1) 

    # translate words used to matrix of numeric values
    X_cnt = cntizer.fit_transform(list_posts)
    # Transfrom previous matrix to term frequency–inverse document frequency i.e. how "important" word is to text
    tfizer = TfidfTransformer()
    # train model for vector fit
    X_tfidf =  tfizer.fit_transform(X_cnt).toarray()

    return X_tfidf
    
def pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True):
    #clean data for text insight
    presonality_types_to_remove_from_posts = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
        'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    
    presonality_types_to_remove_from_posts = [personality.lower() for personality in presonality_types_to_remove_from_posts]

    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()
    #stop words, i.e. A, Any, etc cached for efficiency
    cachedStopWords = stopwords.words("english")

    list_posts = []

    len_data = len(data)
    i=0
    
    for row in data.iterrows():
        i+=1
        if (i % 500 == 0 or i == 1 or i == len_data):
            print("%s of %s rows" % (i, len_data))

        #remove unnecesary data to prepare for NLP
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
            
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t,"")

        list_posts.append(temp)

    list_posts = np.array(list_posts)

    #parts of logic taken from: https://www.kaggle.com/stefanbergstein/byo-tweets-predict-your-myers-briggs-personality
    
    return list_posts

if __name__ == "__main__":
    main()