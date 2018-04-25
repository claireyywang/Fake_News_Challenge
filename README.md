## How to run our code
### The Files and their Uses
-  `extended_setup.py` and `extended_setup_two.py` contain functions to read in data, create vectors for their 
extended feature sets, and save predictions to CSV. They are nearly identical except for a change in output in 
`extended_setup_two.py` that allows it to work with the later extensions. All the extension classifier files call the setup files.
- `extension1.py` augments the feature set of the previous milestones by adding in stopwords and calculating features for
overlap and polarities `extension_mlp.py` runs the published baseline on the augmented feature set. To run the classifier, 
just run `python extension_mlp.py`
-  `extension2.py` applies a 2-step procedure to classify the pairs (first as unrelated/related and second as disagrees/agrees/discusses)
to run its classifier, just run `python extension2.py` in the terminal
-  `extension3.py` applies a RandomForest Classifier on the extended feature set. To run the classifier, run the `python extension3.py` 
-  `extension4.py` applies many different classifiers to the extended feature set. To run the Voting Score classifier 
(the best performing among Extension 4), run `python extension4.py`
