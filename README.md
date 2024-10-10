# football-learning-model
The Football Learning Model (FLM) is a machine learning system trained on https://www.football-data.co.uk/data. FLM designed to predict football match outcomes using historical data.

The training process involves preprocessing the data, which includes historical football match data such as division, date, home team, away team, full-time home and away goals, and other match statistics. The data is split into training and validation sets using k-fold cross-validation. I used different learning rates to find (not) optimal model parameters. The model, which is implemented with PyTorch, is trained over multiple epochs, and features early stopping to prevent overfitting. 

Validation loss and accuracy are monitored to evaluate the model's performance on unseen data. 

The trained model can be used to predict the outcome of individual matches by preparing input features and running sensitivity analysis to understand the impact of each feature on the predictions. (WIP)

If you want to learn more about the methods used in the preprocessing, training, or anything else really I graciously left links for every documentation page I used [below](#documentation-links).



# Definitions that are not useless

## Epoch
An epoch refers to one complete pass through the entire training dataset. During each epoch, the model processes every training example exactly once.

## Validation Loss
Validation loss is a measure of how well the model performs on a separate validation dataset that is not used for training. It is calculated by passing the validation data through the model and computing the loss using a predefined loss function (in this case, criterion). The validation loss helps in monitoring the model's performance on unseen data and is crucial for detecting overfitting.

## Accuracy
Accuracy is indicates the proportion of correctly predicted instances out of the total instances. It is calculated by comparing the model's predictions to the true labels.



# Documentation links

### pandas
Pandas is used for data analysis (obviously), and is by far the best documented module here.
[Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

### numpy
NumPy is like Pandas little brother. Well documented too.
[NumPy Documentation](https://numpy.org/doc/stable/)

### sklearn (Scikit-learn)
Scikit-learn is used for preprocessing.
[Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

### torch (PyTorch)
PyTorch is an optimized tensor library for deep learning. It is extremely well documented, but in light mode.
[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### torch.nn
PyTorch's `torch.nn` is used to define the neural network layers and the loss function. I wish this website was dark mode.
[torch.nn Documentation](https://pytorch.org/docs/stable/nn.html)

### torch.optim
PyTorch's `torch.optim` module is used to define the optimizer for training the neural network model.
[torch.optim Documentation](https://pytorch.org/docs/stable/optim.html)

### torch.utils.data
PyTorch's `torch.utils.data` can create data loaders for batching and shuffling our datasets.
[torch.utils.data Documentation](https://pytorch.org/docs/stable/data.html)

### sklearn.preprocessing
`sklearn.preprocessing` is a really really really well documented module that I used for scaling features and encoding categorical variables.
[sklearn.preprocessing Documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)

### sklearn.model_selection
I used the `sklearn.model_selection` module for k-fold cross-validation, I love how well documented all the Scikit-learn modules are.
[sklearn.model_selection Documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)