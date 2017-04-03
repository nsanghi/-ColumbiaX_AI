import numpy as np
import pandas as pd
import glob

train_path = "./aclImdb/train/" # use terminal to ls files under this directory
test_path = "imdb_te.csv" # test data for grade evaluation

# uncomment this before uploading the file
# train_path = "../resource/asnlib/public/aclImdb/train/" # use terminal to ls files under this directory
# test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
	and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''

    # get texts with +ve sentiments
    comments=[]
    labels =[]
    for file in glob.glob(inpath+"pos/*.txt"):
        text_file = open(file, "r")
        line = text_file.read()
        comments.append(line)
        labels.append(1)
        text_file.close()

    # get texts with -ve sentiments
    for file in glob.glob(inpath+"neg/*.txt"):
        text_file = open(file, "r")
        line = text_file.read()
        comments.append(line)
        labels.append(0)
        text_file.close()

    df = pd.DataFrame.from_dict({'text':comments, 'polarity':labels})
    df = df[['text','polarity']]

    f = open('./stopwords.en.txt', 'r')
    stopwords = list(f)
    stopwords = [w[:-1] for w in stopwords]
    f.close()




    df.to_csv(outpath+name)


if __name__ == "__main__":

    #imdb_data_preprocess(train_path)

    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
  	
    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
     
    '''train a SGD classifier using unigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to unigram.output.txt'''
  	
    '''train a SGD classifier using bigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to unigram.output.txt'''
