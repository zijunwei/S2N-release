import os
import json

user_root = os.path.expanduser('~')
dataset_path = os.path.join(user_root, 'datasets/imagenet12')
fpath = os.path.join(dataset_path, "imagenet_class_index.json")
CLASS_INDEX = None



def decode_predictions(preds, top=5, verbose=False):
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
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)

    if verbose:
        print_results(results)
    return results


def decode_predsngts(preds, targets, top=5, verbose=False):
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
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for target, pred in zip(targets, preds):
        top_indices = pred.argsort()[-top:][::-1]

        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        result.append(tuple(CLASS_INDEX[str(target)]) + (1.,))
        results.append(result)
    if verbose:
        print_results(results)
    return results


def print_results(results):
    print " ---- ---- ---- Batch Results"
    for item_idx, s_result in enumerate(results):
        print " ---- ---- [{:d} | {:d}]".format(item_idx, len(results))
        for idx, s_predict in enumerate(s_result):
            print "{:d}\t{:s}\t{:s}\t{:.04f}".format(idx + 1, s_predict[0], s_predict[1], s_predict[2])


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
            CLASS_INDEX = json.load(open(fpath))
        result = tuple(CLASS_INDEX[str(idx)])

        return result[1]