import re
import random
from torchtext import data



class MyDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, dataset, path=None, examples=None, **kwargs):
        """Create dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        self.examples = examples
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.tokenize = lambda x: clean_str(x).split()
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = "./" if path is None else path
            self.examples = []
            if dataset == 'IMDB':
                with open('IMDB.neg', errors='ignore') as f:
                    examples += [
                        data.Example.fromlist([line, 'negative'], fields) for line in f]
                with open('IMDB.pos', errors='ignore') as f:
                    examples += [
                        data.Example.fromlist([line, 'positive'], fields) for line in f]
            elif dataset == 'financial':
                with open('financial.neg', errors='ignore') as f:
                    self.examples += [
                        data.Example.fromlist([line, 'negative'], fields) for line in f]
                with open('financial.neu', errors='ignore') as f:
                    self.examples += [
                        data.Example.fromlist([line, 'neutral'], fields) for line in f]
                with open('financial.pos', errors='ignore') as f:
                    self.examples += [
                        data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MyDataset, self).__init__(self.examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dataset, validation_ratio=.1, test_ratio=.2, shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of my dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            validation_ratio: The ratio that will be used to get split validation dataset.
            test_ratio: The ratio that will be used to get split test dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        examples = cls(text_field, label_field, dataset, path=None, **kwargs).examples
        if shuffle: random.Random(1).shuffle(examples)
        validation_test_threshold = -1 * int(validation_ratio*len(examples))
        train_validation_threshold = -1 * int((validation_ratio + test_ratio)*len(examples))

        return (cls(text_field, label_field, dataset, examples=examples[:train_validation_threshold]),
                cls(text_field, label_field, dataset, examples=examples[train_validation_threshold:validation_test_threshold]),
                cls(text_field, label_field, dataset, examples=examples[validation_test_threshold:]))
