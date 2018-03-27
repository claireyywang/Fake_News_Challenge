from fnc_feature_engineering import *

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated', 'related']
RELATED = LABELS[0:3]

def generate_features(stances, articles):
  headlines, bodies, labels = [], [], []

  for s in stances:
    labels.append(LABELS.index(s['Stance']))
    headlines.append(s['Headline'])
    bodies.append(articles[s['Body ID']])

  X_overlap = gen_or_load_feats(
      word_overlap_features, headlines, bodies, "features/overlap.npy")
  X_refuting = gen_or_load_feats(
      refuting_features, headlines, bodies, "features/refuting.npy")
  X_polarity = gen_or_load_feats(
      polarity_features, headlines, bodies, "features/polarity.npy")
  X_hand = gen_or_load_feats(
      hand_features, headlines, bodies, "features/hand.npy")

  X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
  return X, labels
