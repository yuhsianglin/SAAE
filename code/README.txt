[Data format]

1. Image data:

For the training and test sets, each has two .npy files: one for features, one for labels.

Feature file is a numpy array with shape (number of training instances, image feature dimension). Suggested to normalized to [0, 1].

Label file is a numpy array with shape (number of training instances, ); i.e. a 1-D numpy array. Labels must be integers and should enumerate as 0, 1, 2, ..., consistent with attribute file (see later).

2. Attribute data:

One .npy file containing attribute vectors (as "self.test_X" in class "dataset"). It should be a numpy array with shape (number of classes, attribute dimension). Suggested to normalized to [0, 1]. Here "number of classes" includes both unseen and seen classes.

The order of the attribute row vectors should be consistent with the labeling in the image label data. For example, the row with index 0 should be the attribute vector of the class with label 0, the row with index 1 should be the attribute vector of the class with label 1, and so on.

3. Unseen classes:

One .npy file of a numpy array with shape (number of unseen class, ); i.e. a 1-D numpy array. It contains the unseen class labels.
