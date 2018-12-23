#!/usr/bin/env python

import nltk, sys, re, string, csv
from nltk.stem.porter import PorterStemmer

#var to toggle print statements
debug = True

########## Helper Functions ##########

#prints only if debug is toggled
def dprint(s):
    if debug:
        print(s)

def read_dataset(file_name):
    #remove \n and \t chars from lines, lowercase and return labels and corresponding sms lists
    with open(file_name) as f:
        lines = [re.sub('[\t]', ' ', line.rstrip('\n')).split(' ', 1) for line in f]
        return lines

def process_dataset(dataset_name):
    print('\n\ndataset Name: \n\n' + dataset_name + '\n\n')
    dataset_contents = read_dataset(dataset_name)

    #lowercase all sms texts and labels and convert list of 2 returned by read_dataset to tuple
    lines = [(line[0].lower(), line[1].lower()) for line in dataset_contents]
    labels = [line[0] for line in lines]
    dprint(labels)

    #word tokenize all sms texts
    tokenized_texts = [nltk.word_tokenize(line[1]) for line in lines]

    #remove stop words
    stopwords = [word.decode('latin1') for word in nltk.corpus.stopwords.words('english')]
    trimmed_texts = [[word for word in line if word.decode('latin1') not in stopwords] for line in tokenized_texts]

    dprint('\n\ntokenized texts:\n')
    dprint(tokenized_texts)
    dprint('\n\ntexts without stopwords:\n')
    dprint(trimmed_texts)

    #total distinct tokens up this point
    all_words = []
    for line in tokenized_texts:
        for word in line:
            all_words += [word]
    unique_tokens = set(all_words)
    dprint("\n\nunique tokens after tokenization: " + str(len(unique_tokens)) + "\n\n")

    #remove punctuation
    punc = [word for word in string.punctuation]
    trimmed_texts = [[word for word in line if word.decode('latin1') not in punc] for line in trimmed_texts]

    dprint('\n\ntexts without stopwords or punctuation:\n')
    dprint(trimmed_texts)

    #stem words
    stemmer = PorterStemmer()
    stemmed_texts = [[(stemmer.stem(word.decode('latin1'))).encode('latin1') for word in line] for line in trimmed_texts]

    dprint('\n\nstemmed texts:\n')
    dprint(stemmed_texts)

    #remove infrequent tokens
    fdist = {}
    for line in stemmed_texts:
        for word in line:
            try:
                if fdist[word]:
                    fdist[word] += 1
            except:
                fdist[word] = 1

    dprint('\n\ntotal frequency distribution:\n')
    dprint(fdist)

    freq_words = [[word for word in line if fdist[word] > 4] for line in stemmed_texts]

    #trim down fdist to the tokens we care about too
    trimmed_fdist = {}
    for key in fdist:
        if fdist[key] > 4:
            trimmed_fdist[key] = fdist[key]

    dprint('\n\nfrequent words (occurance > 4):\n')
    dprint(freq_words)

    dprint('\n\ntrimmed fdist count:' + str(len(trimmed_fdist)) + '\n')

    #make feature vectors
    feature_vectors = []
    for line in freq_words:
        feature_vector = {}
        for word in line:
            try:
                if feature_vector[word]:
                    feature_vector[word] += 1
            except:
                feature_vector[word] = 1
        feature_vectors.append(feature_vector)

    dprint('\n\nfeature vectors:\n')
    dprint(feature_vectors)

    #print feature vectors to .csv file
    if 'train' in dataset_name:
        print_feature_vectors(labels, trimmed_fdist, feature_vectors, 'train')
    elif 'test' in dataset_name:
        print_feature_vectors(labels, trimmed_fdist, feature_vectors, 'test')

def print_feature_vectors(labels, fdist, vectors, dataset):
    #print header row
    with open('HW3_Melkote_'+dataset+'.csv', 'wb') as csvfile:
        file_writer = csv.writer(csvfile, delimiter='\t',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow([word for word in fdist] + ['label'])
        # print('\n\nColumns: ' + str([word for word in fdist]))
        lineNum = 0
        for line in vectors:
            total_line_fdist = []
            for word in fdist:
                try:
                    if line[word]:
                        total_line_fdist += [line[word]]
                except:
                    total_line_fdist += [0]
            file_writer.writerow(total_line_fdist + [labels[lineNum]])
            lineNum+=1

########## Main function ##########

if __name__ == '__main__':
    ########## encoding fix ########## 
    reload(sys)
    sys.setdefaultencoding('utf-8')

    print("\nSMS ham/spam classifier v1.0 by Vikram Melkote and Samer Baslan\n\n")

    #process training set
    process_dataset('train_file_cmps142_hw3')

    #process testing set
    process_dataset('test_file_cmps142_hw3')