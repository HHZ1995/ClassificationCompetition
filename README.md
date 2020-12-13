# Text Classification Competition: Twitter Sarcasm Detection 

## How to test models on your own?
We performed three different models for this competition, machine learning model, BERT-based model and neural-network model. 
You can test whichever you like by following the instructions respectively. 
## Machine Learning Model
***Pre-requisite:***
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

***Run the script:***
1. Clone the repository to your computer
2. cd ClassificationCompetition/ML
3. Run python **TFIDF_RandomForests.py**
4. Output prediction result (**anser.txt**) will be able to find in **ClassificationCompetition** folder

## BERT-base-uncased Model
***Pre-requisite:***
- Tensorflow
- Transformer
- PyTorch
- torchtext
- BERT
- cuda toolkit https://anaconda.org/anaconda/cudatoolkit (Hardware requirments: https://docs.anaconda.com/anaconda/user-guide/tasks/gpu-packages/) 
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

***Run the script:***
1. Clone the **BERT-base-uncase-code** into your local computer (all the dataset is already prepared for this model)
2. Run python **BERT_Model.py**
(If you see this message in the console "Running this sequence through the model will result in indexing errors", this is just the warning, NOT actual error!)
3. The final models (**metrics.pt** and **model.pt**) and output prediction result (**anser.txt**) will be able to find in **result** folder

## CNN+LSTM+DNN Model
***Pre-requisite:***
- nltk (TweetTokenizer)
- Keras
- Tensorflow
- numpy
- scipy
- gensim (if you are using word2vec)
- itertools
- sklearn

***Run the script:***
1. Clone the repository
2. Download following files from the link - https://drive.google.com/drive/folders/0B7C_0ZfEBcpRbDZKelBZTFFsV0E?usp=sharing, to the following directory: ClassificationCompetition/resource/text_model/weights
   - GoogleNews-vectors-negative300.bin
   - model.jsonhdf5
   - weights.05__.hdf5
3. cd ClassificationCompetition/CNN_LSTM_DNN
4. run Python sarcasm_detection_model_CNN_LSTM_DNN.py
