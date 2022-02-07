# MNE

This is the example of the experiment using preprocessing methods from the `MNE` library and classification methods from the `Keras` and the `scikit-learn` libraries for EEG data stored in the `BrainVision` format. The used dataset in the `BrainVision` format, which is located in the _PROJECT_DAYS_P3_NUMBERS_ folder, is described in this [article](https://www.nature.com/articles/sdata2016121)

## Used preprocessing methods:
- low-pass and high-pass filtering
- epoch extraction
- baseline correction
- artifact removal with the peak-to-peak amplitude rejection
- windowed means feature extraction

## Used classifiers:
- Convolutional neural network
- LSTM neural network
- Linear discriminant analysis
- Support vector machines
- Logistic regression
- Random Forest Classifier
- XGBoost

## Run:
The program is executable from the command line using the _main.py_ file with one argument represents the choice of classifier. The command has the following form:

`python main.py <classifier>`

The user can choose from 7 types of classifiers, so the possible variants of the command are:
- `python main.py lda`
- `python main.py svm`
- `python main.py cnn`
- `python main.py rnn`
- `python main.py lor`
- `python main.py xgb`
- `python main.py randfor`

