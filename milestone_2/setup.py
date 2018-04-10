# imports
from csv import DictReader
from csv import DictWriter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

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
def pipeline_train(train, test, lim_unigram):
    """
    Process train set, create relevant vectorizers
    
    Args:
        train: FNCData object, train set
        test: FNCData object, test set
        lim_unigram: int, number of most frequent words to consider

    Returns:
        train_set: list, of numpy arrays
        train_stances: list, of ints
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()
    """


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
    test = []
    headers_track = {}
    bodies_track = {}
    cosines = {}

    # iterate over each header/body (id)
    for instance in test.instances:
    	header = instance['Headline']
    	body_id = instance['Body ID']
    	# get tf and tfidf for header
    	if header not in headers:
    		header_count = bow_vectorizer.transform([header]).toarray()
    		header_tf = tfreq_vectorizer.transform(header_count).toarray()[0].reshape(1, -1)
    		header_tfidf = tfidf_vectorizer.transform([header]).toarray().reshape(1, -1)
            headers_track[header] = (header_tf, header_tfidf)
        else:
        	header_tf = headers_track[header][0]
            header_tfidf = headers_track[header][1]

        # get tf and tfidf for body
        if body_id not in bodies_track:
        	body_count = bow_vectorizer.transform([test.bodies[body_id]]).toarray()
        	body_tf = tfreq_vectorizer.transform(body_count).toarray()[0].reshape(1, -1)
        	body_tfidf = tfidf_vectorizer.transform([test.bodies[body_id]]).toarray().reshape(1, -1)
            bodies_track[body_id] = (body_tf, body_tfidf)
        else:
        	body_tf = bodies_track[body_id][0]
            body_tfidf = bodies_track[body_id][1]

        # get the cosine similarity for header and body tfidf
        if (header, body_id) not in cosines:
        	tfidf_cos = cosine_similarity(header_tfidf, body_tfidf)[0].reshape(1, 1)
            cosines[(head, body_id)] = tfidf_cos
        else:
        	tfidf_cos = cosines[(head, body_id)]

        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        test.append(feat_vec)
        
    return test

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
