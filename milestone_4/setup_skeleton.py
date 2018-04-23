# imports
from csv import DictReader
from csv import DictWriter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initializing global data structures
stance_label = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
stance_label_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

# Define data class
class FNCData:

    """
    Define class for Fake News Challenge data
    """
    def __init__(self, file_instances, file_bodies):
        self.headers = {}
        self.bodies = {}
        bodies = self.read(file_bodies)
        self.instances = self.read(file_instances)

        for instance in self.instances:
            if instance['Headline'] not in self.headers:
                header_id = len(self.headers)
                self.headers[instance['Headline']] = header_id
            instance['Body ID'] = int(instance['Body ID'])

        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['articleBody']

    def read(self, filename):
        """
        Read Fake News Challenge data from CSV file
        Args:
            filename: str, filename + extension
        Returns:
            rows: list, of dict per instance
        """
        rows = []

        with open(filename, "r", encoding='utf-8') as fl:
            reader = DictReader(fl)
            # dict reader stores a mapping for headlines as well as body
            for line in reader:
                rows.append(line)

        return rows

# Define relevant functions
def create_vectors(train, dev, test, lim_unigram):
    """
    Create relevant vectorizers
    The paper used different vocabularies for the TF and TF-IDF vectors -
    the TF contained the 5,000 most common words in the train set, minus
    the stop words. The TF-IDF contained the 5,000 most common words in the
    train + test set, minus the stop words. We modify this approach here to
    include the train + test + dev set for the TF-IDF vocabualry.

    Args:
        train: FNCData object, train set
        dev: FNCData object, dev set
        test: FNCData object, test set
        lim_unigram: int, number of most frequent words to consider

    Returns:
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()
    """

    # should finally contain the unique heads and bodies for each data set.
    heads_train = []
    heads_dev = []
    heads_test = []

    bodies_train = []
    bodies_dev = []
    bodies_test = []

    # use these dicts to ensure that you do not add the same head or body to different data sets
    heads_unique = {}
    bodies_unique_id = {}

    """
    TODO: Populate the arrays and dicts above with the appropriate heads and bodies.
          Remember - you can iterate through the instances from each FNCData object!

    """

    # create vectorizers using the appropriate data set(s)
    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads_train + bodies_train)

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    # tfreq = tfreq_vectorizer.transform(bow).toarray()

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).fit(heads_train + bodies_train + heads_dev + bodies_dev + heads_test + bodies_test)

    return (bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)


def pipeline_train(train, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    """
    Process train set, return feature vector for neural net

    Args:
        train: FNCData object, train set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    Returns:
        train_set: list, of numpy arrays
        train_stances: list, of ints
    """

    heads_unique = {}
    bodies_unique_id = {}
    cos_unique = {}

    train_X = []
    train_y = []

    for instance in train.instances:

        # track the true classification label
        train_y.append(stance_label[instance['Stance']])

        head = instance['Headline']
        b_id = instance['Body ID']

        """
        TODO: Create the TF and TF-IDF vectors for heads and bodies, checking heads_unique
              and bodies_unique_id to ensure that there are no redundant calculations.

        """

        """
        TODO: Calculate the cosine similarity of the head and body TF-IDF vectors,
              checking cos_unique to ensure that there are no redundant calculations.

        """
        # create an appropriate vector from the head and body TF vectors (and TF-IDF cosine
        # similarity) to add to the training set. Note this line will not compile until you
        # define the head_tf, body_tf, and tfidf_cos variables above.
        vect = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        train_X.append(vect)

    return train_X, train_y


def pipeline_dev(dev, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    """
    Process dev set, return feature vector for neural net

    Args:
        dev: FNCData object, dev set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    Returns:
        dev_set: list, of numpy arrays
        dev_stances: list, of ints
    """

    heads_unique = {}
    bodies_unique_id = {}
    cos_unique = {}

    dev_X = []
    dev_y = []

    for instance in dev.instances:

        # track the true classification label
        dev_y.append(stance_label[instance['Stance']])

        head = instance['Headline']
        b_id = instance['Body ID']

        """
        TODO: Create the TF and TF-IDF vectors for heads and bodies, checking heads_unique
              and bodies_unique_id to ensure that there are no redundant calculations.

        """

        """
        TODO: Calculate the cosine similarity of the head and body TF-IDF vectors,
              checking cos_unique to ensure that there are no redundant calculations.

        """
        # create an appropriate vector from the head and body TF vectors (and TF-IDF cosine
        # similarity) to add to the dev set. Note this line will not compile until you
        # define the head_tf, body_tf, and tfidf_cos variables above.

        vect = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        dev_X.append(vect)

    return dev_X, dev_y

def pipeline_test(test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    """
    Process test set
    Args:
        test: FNCData object, test set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()
    Returns:
        test_set: list, of numpy arrays
    """

    heads_unique = {}
    bodies_unique_id = {}
    cos_unique = {}

    test_X = []

    # iterate over each header/body (id)
    for instance in test.instances:

        head = instance['Headline']
        b_id = instance['Body ID']

        """
        TODO: Create the TF and TF-IDF vectors for heads and bodies, checking heads_unique
              and bodies_unique_id to ensure that there are no redundant calculations.

        """

        """
        TODO: Calculate the cosine similarity of the head and body TF-IDF vectors,
              checking cos_unique to ensure that there are no redundant calculations.

        """
        # create an appropriate vector from the head and body TF vectors (and TF-IDF cosine
        # similarity) to add to the test set. Note this line will not compile until you
        # define the head_tf, body_tf, and tfidf_cos variables above.

        vect = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        test_X.append(vect)

    return test_X

def load_model(sess):
    """
    Load TensorFlow model
    Args:
        sess: TensorFlow session
    """
    saver = tf.train.Saver()
    saver.restore(sess, './model/model.checkpoint')


def save_predictions(pred, file):
    """
    Save predictions to CSV file
    Args:
        pred: numpy array, of numeric predictions
        file: str, filename + extension
    """

    with open(file, 'w') as csvfile:
        fieldnames = ['Stance']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for instance in pred:
            writer.writerow({'Stance': stance_label_rev[instance]})
