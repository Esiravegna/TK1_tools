import pandas as pd
from keras.preprocessing.image import load_img

import logging
logger = logging.getLogger(__name__)

class TK1CorpusBuilderError(Exception):
    pass


class TK1ImageCorpusGenerator(object):
    """
    Just a corpus builder with a minimum memory footprint
    Essentially, it reads a csv wich should contain at least three columns:
        - ITEM ID
        - Label to predict
        - Relative to the file provided path for the actual image
    """
    def __init__(self, path_to_file,
                 batch_size=32,
                 skip_header=False,
                 column_id=0,
                 column_label=1,
                 column_path=2):
        """
        Constructor. Just reads the file and creates the two lists to be used.
        :param path_to_file: Where the file resides
        :param batch_size: how many image to return as per minibatch
        :param skip_header: Does the file have a header?
        :param column_id: Column number on where the item ID resides
        :param column_label: Column number on where the label is stored
        :param column_path: Column number to get the relative path
        """
        try:
            if skip_header:
                corpus_df = pd.read_csv(path_to_file, header=None)
            else:
                corpus_df = pd.read_csv(path_to_file)
        except OSError:
            raise TK1CorpusBuilderError("{} not found".format(path_to_file))

        # let' shuffle the corpus
        corpus_df = corpus_df.sample(frac=1).reset_index(drop=True)

        #allright, let' store the lists then
        self.batch_size = batch_size
        self.ids = corpus_df[corpus_df.columns[column_id]].values
        self.labels = corpus_df[corpus_df.columns[column_label]].values
        self.image_path = corpus_df[corpus_df.columns[column_path]].values

        # we'e done here, deleting stuff
        del corpus_df

    def __len__(self):
        """
        Just a magic method for getting the length of the array
        :return:
        """
        return len(self.ids)

    def __iter__(self):
        """
        the corpus iterator
        :return: a self.batch_size data, labels from the list with the batch size as chunck
        """
        for i in range(0, len(self.ids), self.batch_size):
            yield [a for a in map(load_img, self.image_path[i:i + self.batch_size])], self.labels[i:i + self.batch_size]


