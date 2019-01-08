"""
This content is released under the GNU License http://www.gnu.org/copyleft/gpl.html
Author: Marina Danchovsky Ibrishimova
Purpose: A prototype for an incident classification system, which uses a machine learning model to determine the probability that an event is an incidentself.

Version: 1.0

The prototype is composed of 3 modules:
The first module has the following functions:
- generates a larger dataset from a sample dataset of events using Python’s numpy choice function. (only needed if a large dataset is not present)
- transforms the features of the dataset into the proper format that subsequent modules can use. (Only needed to verify the fitness of different combinations of features as proposed in this paper)

The second module contains a logistic regression model as described in [1]. This one has 3 more features

The third module contains the interactive incident report form, which is needed to obtain information about the event focuses on the event information gathering by
- asking for the event description and running this description through Google NLP API in order to obtain a sentiment analysis score and magnitude, which are the first two features that the model needs in order to make a prediction on whether the event is an incident or not.
- asking if it is known whether the event affects more than a certain number of units, whether it attacks the confidentiality, or integrity, or availability of data and services, and whether the event can be confirmed or not. These are the last 3 features the model needs in order to make a prediction on whether the event is an incident or not
- feeding the collected features to the model, which returns a probability that the event is an incident

References:
[1] Logistic Regression: Calculating a probability, www.developers.google.com/machine-learning/crash-course/logistic-regression/calculating-a-probability, Retrieved 02.06.2018

"""
import os
#replace with your own application credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="location_of_your_google_credentials"
import math
from IPython import display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.python.saved_model import tag_constants
# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
"""Prepare data set.
    1. Generate synthetic data from sample with same distribution for supervised
    learning where each example in the dataset consists of a set number of features
    and a target/label indicating whether the example is an incident or not
    2. encode features' values for easier testing of which set of features
    is optimal for learning
"""
df = pd.read_csv("nnt.csv", sep=",")
size_of_sample = 20000
#generate sample synthetic data with the same distribution as the original dataset.
rows = np.random.choice(df.index.values, size_of_sample)
incident_dataframe = df.loc[rows]

"""Found the best set of features using GA:
   sentiment,magnitude,affects_more_than_x,
   impacts_cia,is_confirmed,is_incident
"""
chromosome=[1,0,1,1,1,1,1,1,1]
#function to encode features's values
def select_features(chromosome,incident_dataframe):
    i = 0
    selected = []
    for x in chromosome:
        if x == 1:
            selected.append(incident_dataframe.columns.get_values()[i])
        i = i + 1
    return selected

selected = select_features(chromosome,incident_dataframe)

def shortans(ans):
    if ans == "yes":
        ans = 1.0
    elif ans == "no":
        ans = 0
    else:
        ans = 0.5
    return float(ans)

def preprocess_features(incident_dataframe):
  """Prepares input features

  Args:
    incident_dataframe: A Pandas DataFrame expected to contain data
      from the risidata data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = incident_dataframe[selected]
  processed_features = selected_features.copy()
  return processed_features

def preprocess_targets(incident_dataframe):
  """Prepares targets (i.e., labels) from data set.
    We need this because our model is utilizing supervised learning
  Args:
     dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  output_targets["event_breaks_law"] = incident_dataframe["is_incident"];
  return output_targets

# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(incident_dataframe.head(12000))
training_targets = preprocess_targets(incident_dataframe.head(12000))
# Choose the last 8000 (out of 20000) examples for validation.
validation_examples = preprocess_features(incident_dataframe.tail(8000))
validation_targets = preprocess_targets(incident_dataframe.tail(8000))

#Double-check that we've done the right thing.
print ("Training examples summary:")
display.display(training_examples.describe())
print ("Validation examples summary:")
display.display(validation_examples.describe())

print ("Training targets summary:")
display.display(training_targets.describe())
print ("Validation targets summary:")
display.display(validation_targets.describe())

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.
  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}
    if targets is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, targets)
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices(inputs) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)


    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(1000)

    # Return the next batch of data.
    return ds.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

  """Trains a linear classification model.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `incident_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `incident_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `incident_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `incident_dataframe` to use as target for validation.

  Returns:
    A `LinearClassifier` object trained on the training data."""


  periods = 10
  steps_per_period = steps / periods

  # Create a linear classifier object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )

  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples,
                                          training_targets["event_breaks_law"],
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                  training_targets["event_breaks_law"],
                                                  num_epochs=1,
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                    validation_targets["event_breaks_law"],
                                                    num_epochs=1,
                                                    shuffle=False)

  # Train the model
  print ("Training model...")
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  #print("LogLoss (on training data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    """validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)"""

    # Get just the probabilities for the positive class.
    """validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
    plt.plot(false_positive_rate, true_positive_rate, label="our model")
    plt.plot([0, 1], [0, 1], label="random classifier")
    _ = plt.legend(loc=4)
    plt.show()"""

  print ("Model training finished.")
  evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

  # Output a graph of loss metrics over periods.
  """plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()
  plt.show()"""

  print ("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
  print ("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
  print ("Precision on the validation set: %0.2f" % evaluation_metrics['precision'])
  print ("Recall on the validation set: %0.2f" % evaluation_metrics['recall'])

  return linear_classifier

# Train all the model!
linear_classifier = train_linear_classifier_model(
    learning_rate=0.05,
    steps=500,
    batch_size=50,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

# Instantiate all the client!
client = language.LanguageServiceClient()

# Collect all the data!
print("State the nature of your emergency.")

# The text to analyze
no_of_lines = 100
text = input("")

document = types.Document(
    content=text,
    type=enums.Document.Type.PLAIN_TEXT)

# Detects the sentiment of the text
sentiment_anal = client.analyze_sentiment(document=document).document_sentiment

print('Sentiment: {}, {}'.format(sentiment_anal.score, sentiment_anal.magnitude))
sentiment= float(sentiment_anal.score)
magnitude= float(sentiment_anal.magnitude)
tokens = client.analyze_syntax(document).tokens

# part-of-speech tags from enums.PartOfSpeech.Tag
pos_tag = ('UNKNOWN', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM',
               'PRON', 'PRT', 'PUNCT', 'VERB', 'X', 'AFFIX')
nouns = ""
entities = ""
verbs = ""
adj = ""
for token in tokens:
    if(pos_tag[token.part_of_speech.tag] == 'NOUN'):
        nouns = nouns + " " + token.text.content
    elif(pos_tag[token.part_of_speech.tag] == 'VERB'):
        verbs = verbs + " " + token.text.content
    elif(pos_tag[token.part_of_speech.tag] == 'ADJ'):
        adj = adj + " " + token.text.content
    #print(u'{}: {}'.format(pos_tag[token.part_of_speech.tag],token.text.content))
if nouns != "":
    docnouns = types.Document(
        content=nouns,
        type=enums.Document.Type.PLAIN_TEXT)
    sentiment_nouns = client.analyze_sentiment(document=docnouns).document_sentiment
    sentiment_nouns = sentiment_nouns.score
else:
    sentiment_nouns = 0

if verbs != "":
    docverbs = types.Document(
        content=verbs,
        type=enums.Document.Type.PLAIN_TEXT)
    sentiment_verbs = client.analyze_sentiment(document=docverbs).document_sentiment
    sentiment_verbs = sentiment_verbs.score
else:
    sentiment_verbs = 0

if adj != "":
    docadj = types.Document(
        content=adj,
        type=enums.Document.Type.PLAIN_TEXT)
    sentiment_adj = client.analyze_sentiment(document=docadj).document_sentiment
    sentiment_adj = sentiment_adj.score
else:
    sentiment_adj = 0

print(nouns)
print(sentiment_nouns)
print(verbs)
print(sentiment_verbs)
print(adj)
print(sentiment_adj)

print("Does the event cause damage?")

attacks_cia = input("")
attacks_cia = shortans(attacks_cia)
print("Does the event affect more than one? ")

affects_more_than_x = input("")
affects_more_than_x = shortans(affects_more_than_x)
print("Can you prove that the event took place?")

is_confirmed = input("")
is_confirmed = shortans(is_confirmed)

# My other code is cleaner than this I swear
predict_unlabeled = {}
predict_unlabeled["nouns"] = [sentiment_nouns]
predict_unlabeled["verbs"] = [sentiment_verbs]
predict_unlabeled["adjectives"] = [sentiment_adj]
predict_unlabeled["sentiment"] = [sentiment]
predict_unlabeled["magnitude"] = [magnitude]
predict_unlabeled["affects_more_than_x"] = [affects_more_than_x]
predict_unlabeled["impacts_cia"] = [attacks_cia]
predict_unlabeled["is_confirmed"] = [is_confirmed]


# Make an unlabeled prediction using the features above
unlabeled_predictions = linear_classifier.predict(input_fn=lambda:eval_input_fn(predict_unlabeled,labels=None,batch_size=1))
# Returns a probability that an event is an incident
unlabeled_predictions = np.array([item['probabilities'][1] for item in unlabeled_predictions])
print("The probability that the event is an incident is:")
print(unlabeled_predictions)
