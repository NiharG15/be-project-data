''' Multithreaded NLP Tasks '''
from multiprocessing import Pool
import itertools as it
import numpy as np
from tqdm import tqdm
import spacy
import pandas as pd

nlp = None

def preprocess(nlp, doc):
    """ Preprocess given doc """
    return nlp(' '.join([token.text for token in doc[2:] if token.pos_ in ['ADJ', 'NOUN', 'VERB', 'ADV']]))

def get_similarity(row):
    """
    Return the similarity between the strings contained in the array strings

        nlp: spaCy nlp object
        strings: String array

        returns: similarity score
    """
    strings = row.qa
    if isinstance(strings, str):
        strings = eval(strings)
    similarity = list()
    for str1, str2 in it.combinations(strings, 2):
        doc1 = preprocess(nlp, nlp(str1))
        doc2 = preprocess(nlp, nlp(str2))
        similarity.append(doc1.similarity(doc2))
    avg_similarity = np.mean(similarity)
    return [row.folder, row.file, strings, avg_similarity]

def init():
    global nlp
    nlp = spacy.load('en')

def run_similarity_task(df):
    """
    Calculate similarity using multiple processes

        nlp: spaCy nlp object,
        df: dataframe with columns = (folder, file, qa)
    """
    similarity_data = list()

    pbar = tqdm(total=df.shape[0])

    def cb(result):
        pbar.update()
        similarity_data.append(result)

    def ecb(err):
        print(err)

    pool = Pool(processes=8, initializer=init)

    for _, row in df.iterrows():
        pool.apply_async(func=get_similarity, args=(row,), callback=cb, error_callback=ecb)

    pool.close()
    pool.join()

    return similarity_data

def run_task():
    base_path = '/run/media/niharg/DATA/datasets/ads/'

    df = pd.read_pickle(base_path + 'annotations_images/qa_combined2_cleaned1.pickle')

    for i in range(0, 10): # Iterate over folders
        print('Current Folder: {}'.format(i))
        sub_df = df.loc[df.folder == str(i), :]
        sim_data = run_similarity_task(sub_df)
        sim_data_df = pd.DataFrame(sim_data, columns=['folder', 'file', 'qa', 'similarity'])
        sim_data_df.to_pickle('{}{}/similarity/sim_data_df_{}.pickle'.format(base_path, 'annotations_images', i))
        print('Saved File: {}'.format('{}{}/sim_data_df_{}.pickle'.format(base_path, 'annotations_images', i)))


def main():
    run_task()

if __name__ == '__main__':
    main()