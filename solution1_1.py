from datasets import load_dataset
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import torch
from torch import nn
import time

class NNModel(nn.Module):
    def __init__(self, hidden_size1 = 2048, hidden_size2 = 512, hidden_size3 = 64):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(5000, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 5)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

def calculate_correct_tag_num(prediction,y):
    prediction = torch.max(prediction,1)[1]
    correct = 0
    for i,j in zip(prediction,y):
        if i==j:
            correct+=1
    return correct


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    tot_loss = 0
    accuracy = 0
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X.float().to(device))
        loss = loss_fn(pred, y.type(torch.LongTensor).to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        accuracy +=calculate_correct_tag_num(pred, y)
    print("loss: ", tot_loss/size)
    print("Accuracy: ", accuracy/size)

def eval_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    tot_loss = 0
    accuracy = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.float().to(device))
        loss = loss_fn(pred, y.type(torch.LongTensor).to(device))
        
        tot_loss += loss.item()
        accuracy +=calculate_correct_tag_num(pred, y)
    print("loss: ", tot_loss/size)
    print("Accuracy: ", accuracy/size)

def bag_of_words():
    data = load_dataset("amazon_reviews_multi", "de")

    train = data['train'].shuffle().select(range(30000))
    validation = data['validation'].shuffle().select(range(2000))
    test = data['test'].shuffle().select(range(2000))

    train_review_body = []
    train_stars = []

    valid_review_body = []
    valid_stars = []

    test_review_body = []
    test_stars = []

    def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):

        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

        ## Tokenize ##
        lst_text = text.split()
        ## remove Stopwords
        if lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in 
                        lst_stopwords]

        ## Stemming ##
        if flg_stemm == True:
            ps = nltk.stem.porter.PorterStemmer()
            lst_text = [ps.stem(word) for word in lst_text]

        ## Lemmatization ##
        if flg_lemm == True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            lst_text = [lem.lemmatize(word) for word in lst_text]

        ## back to string from list ##
        text = " ".join(lst_text)
        return text

    for json_doc in train:
        train_review_body.append(utils_preprocess_text(json_doc['review_body']))
        train_stars.append(json_doc['stars']-1)

    for json_doc in validation:
        valid_review_body.append(utils_preprocess_text(json_doc['review_body']))
        valid_stars.append(json_doc['stars']-1)

    for json_doc in test:
        test_review_body.append(utils_preprocess_text(json_doc['review_body']))
        test_stars.append(json_doc['stars']-1) # class - 0,1,2,3,4

    # Bag of words #
    vectorizer_bow = CountVectorizer(max_features=5000, ngram_range=(1,2))

    # BOW #
    vectorizer_bow.fit(train_review_body)
    X_train = vectorizer_bow.transform(train_review_body)
    X_val = vectorizer_bow.transform(valid_review_body)
    x_test = vectorizer_bow.transform(test_review_body)
    bow_vocabulary = vectorizer_bow.vocabulary_

    X_train = csr_matrix(X_train).toarray()
    X_val = csr_matrix(X_val).toarray()
    x_test = csr_matrix(x_test).toarray()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = NNModel().to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(train_stars))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)

    dev_data = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.Tensor(valid_stars))
    dev_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=64)

    test_data = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(test_stars))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)


    epochs = 20
    for t in range(epochs):
        start_t = time.time()
        print("Epoch : ", t+1)
        print("Train:")
        train_loop(train_dataloader, net, loss_function, optimizer, device)
        print("Validation:")
        eval_loop(dev_dataloader, net, loss_function, device)
        print("Time taken: ",time.time() - start_t)

    print("__________________________________________________")
    print("Test:")
    eval_loop(test_dataloader, net, loss_function, device)