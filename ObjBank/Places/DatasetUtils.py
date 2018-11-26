import os, sys
import json
import shutil

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')

CLASS_INDEX = None
file_name = 'categories_places365.txt'

LABEL_MAP_PATH = os.path.join(project_root, 'dev_kits', file_name)



def decode_predictions(preds, top=5, verbose=False):
    global  CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 365:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 365)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        if not os.access(LABEL_MAP_PATH, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        shutil.move(file_name, LABEL_MAP_PATH)
        classes = list()
        with open(LABEL_MAP_PATH) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        CLASS_INDEX = tuple(classes)
    results = []
    for pred in preds:

          top_indices = pred.argsort()[-top:][::-1]
          result = [tuple([CLASS_INDEX[i], pred[i]]) for i in top_indices]
          result.sort(key=lambda x:x[1], reverse=True)
          results.append(result)
    if verbose:
        print_results(results)
    return results



def print_results(results):
    print " ---- ---- ---- Batch Results"
    for item_idx, s_result in enumerate(results):
        print " ---- ---- [{:d} | {:d}]".format(item_idx+1, len(results))
        for idx, s_predict in enumerate(s_result):
            print "{:d}\t{:s}\t{:.04f}".format(idx + 1, s_predict[0], s_predict[1])

def get_tags(idx):
    """Decodes the prediction of an ImageNet model.
    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.
    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    global CLASS_INDEX

    if CLASS_INDEX is None:
        if not os.access(LABEL_MAP_PATH, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
            shutil.move(file_name, LABEL_MAP_PATH)
        classes = list()
        with open(LABEL_MAP_PATH) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        CLASS_INDEX = tuple(classes)

    result = (CLASS_INDEX[(idx)])

    return result