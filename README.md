# Text Classification Competition: Twitter Sarcasm Detection 

## Dataset format:

Each line contains a JSON object with the following fields : 
- ***response*** :  the Tweet to be classified
- ***context*** : the conversation context of the ***response***
	- Note, the context is an ordered list of dialogue, i.e., if the context contains three elements, `c1`, `c2`, `c3`, in that order, then `c2` is a reply to `c1` and `c3` is a reply to `c2`. Further, the Tweet to be classified is a reply to `c3`.
- ***label*** : `SARCASM` or `NOT_SARCASM` 

- ***id***:  String identifier for sample. This id will be required when making submissions. (ONLY in test data)

For instance, for the following training example : 

`"label": "SARCASM", "response": "@USER @USER @USER I don't get this .. obviously you do care or you would've moved right along .. instead you decided to care and troll her ..", "context": ["A minor child deserves privacy and should be kept out of politics . Pamela Karlan , you should be ashamed of your very angry and obviously biased public pandering , and using a child to do it .", "@USER If your child isn't named Barron ... #BeBest Melania couldn't care less . Fact . 💯"]`

The response tweet, "@USER @USER @USER I don't get this..." is a reply to its immediate context "@USER If your child isn't..." which is a reply to "A minor child deserves privacy...". Your goal is to predict the label of the "response" while optionally using the context (i.e, the immediate or the full context).

***Dataset size statistics*** :

| Train | Test |
|-------|------|
| 5000  | 1800 |

For Test, we've provided you the ***response*** and the ***context***. We also provide the ***id*** (i.e., identifier) to report the results.

***Submission Instructions*** : Please add a comma separated file named `answer.txt` containing the predictions on the test dataset. The file should have no headers and have exactly 1800 rows. Each row must have the sample id and the predicted label. For example:

twitter_1,SARCASM  
twitter_2,NOT_SARCASM  
...


## BERT-base-uncased model
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

***Running the script:***
1. Clone the **BERT-base-uncase-code** into your local computer (all the dataset is already prepared for this model)
2. Run python **BERT_Model.py**
(If you see this message in the console "Running this sequence through the model will result in indexing errors", this is just the warning, NOT actual error!)
3. The final models (**metrics.pt** and **model.pt**) and output prediction result (**anser.txt**) will be able to find in **result** folder

