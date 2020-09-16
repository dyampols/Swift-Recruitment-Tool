import re
import numpy as np
import collections
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

binary_dict_1 = {"I": 0, "E": 1}
binary_dict_2 = {"N": 0, "S": 1}
binary_dict_3 = {"T": 0, "F": 1}
binary_dict_4 = {"J": 0, "P": 1}


def main():
    text = input("Please enter text to dermine personality type of individual: ")
    list_posts = np.genfromtxt('list_posts.csv',dtype = str,delimiter=',')
    string_result = ""
    binary_vals = ['I/E','N/S','T/F','J/P']
    for binary in binary_vals:
        index = str(binary_vals.index(binary))
        print("Determining type " + binary + "...")
        binary_result = make_prediction(text,binary,list_posts,index)
        translated = translate_binary_result(binary_result,binary)
        string_result += translated
    print("The personality type of the individual is: " + string_result)

def translate_binary_result(binary_result,binary):
    if binary == 'I/E':
        if binary_result[0] == 0:
            return "I"
        else:
            return "E"
    elif binary == 'N/S':
        if binary_result[0] == 0:
            return "N"
        else:
            return "S"
    elif binary == 'T/F':
        if binary_result[0] == 0:
            return "T"
        else:
            return "F"
    elif binary == 'J/P':
        if binary_result[0] == 0:
            return "J"
        else: 
            return "P"
        
def vectorize_posts(list_posts):
    # Posts to a matrix of token counts
    cntizer = CountVectorizer(analyzer="word", 
                                max_features=1500, 
                                tokenizer=None,    
                                preprocessor=None, 
                                stop_words=None,  
                                max_df=0.7,
                                min_df=0.1) 

    # Learn the vocabulary dictionary and return term-document matrix
    X_cnt = cntizer.fit_transform(list_posts)

    # Transform the count matrix to a normalized tf or tf-idf representation
    tfizer = TfidfTransformer()

    # Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
    X_tfidf =  tfizer.fit_transform(X_cnt).toarray()

    return X_tfidf

def make_prediction(text,binary,list_posts,index):
    filename = 'finalized_model'+index+'.sav'
    logregg = pickle.load(open(filename, 'rb'))
    data= [[len(text.split()),text.count('?'),text.count('...'),text.count('!'),text.count('music'),text.count('http'),text.count('jpg')+ text.count('image'),text.count(':)') + (text.count(':(') + text.count(':O') + text.count(';)') + text.count(':P') + text.count(':''(') + text.count(':-D'))]]
    data_frame = pd.DataFrame(data, columns = ['words_per_comment','question_per_comment','ellipsis_per_comment','excl_per_comment','music_per_comment','http_per_comment','img_per_comment','emojis_per_comment']) 
    a = np.array([text])
    appended_text_to_be_processed = np.append(list_posts,a)
    vectorized_posts = vectorize_posts(appended_text_to_be_processed)

    extracted_vectorized_posts = vectorized_posts[8674,:]

    for i in range(len(extracted_vectorized_posts)):
        data_frame['text'+str(i)] = extracted_vectorized_posts[i]

    preddicted_binary = logregg.predict(data_frame)
    return preddicted_binary

if __name__ == "__main__":
    main()