pytorch_intro

**tensors**
- specialized data structure that is similar to arrays and matrices
- used to encode the inputs and outputs of a model & a model's parameters
- tensors can run on GPUs/other hardware accelerators 

**bag of words**
- common method to represent text as numbers - so the ML model can work with it
- each n-gram gets a TF-IDF score based on TF and IDF
  - TF-IDF: 
    - TF: frequency in document (a higher frequency suggests greater importance)
    - IDF: frequency across documents (a lower frequency/if a term appears in fewer documents, it is more likely to be meaningful and specific)


**goal**
predict whether a sentence is written in English or Spanish


# 1. set up data
``` python 

#1. import - option 1 
import torch #imports the core pytorch library; needed for mathematical operations
import torch.nn as nn #imports neural network module: contains layers, activiation functions, loss functions, etc. 
from torch.utils.data import DataLoader #for large datasets; batches the data (e.g. 64 samples at a time) & shuffles data (so training isn't biased)
from torchvision import datasets #provides ready-to-use popular datasets 
from torchvision.transforms import ToTensor #converts image data to pytorch tensors 

#1. import - option 2
- #DataLoader and datasets excluded when using a small dataset
import torch
import torch.nn as nn
import torch.nn.functional as F #imports the functional API for neural networks (i.e. F.softmax activation function, loss function); used to apply mathematical operations w/o creating an object 
import torch.optim as optim #imports optimizers (e.g. Adam)

torch.manual_seed(123) #set random need for pytorch's random number generator 

#2. load data
data = [("me gusta comer en la cafeteria".split(), "SPANISH"), #.split() by default splits when there is whitespace 
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}


#3. assign each unique word to an index 

# Function to map each word in the vocab to an unique integer
# Indexing the Bag of words vector

word_to_ix = {} #create empty dictionary; key = words (strings) | values = unique integers
for sent, _ in data + test_data: #for each sentence in data & test data (_ says to ignore the value, since data and test_data are dictionaries)
    for word in sent: #for each word in each sentence 
        if word not in word_to_ix: #if word is not in the dictionary
            word_to_ix[word] = len(word_to_ix) #give the word an index based on the current length of the dictionary (i.e. first word gets index 0; 2nd unique word gets index 1)
print(word_to_ix) #dict with words and indices

#4. define vocab size & number of labels
VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2 #bc have 2 labels: English and Spanish

```

# 2. build a classifier 
- every model is a subclass of `nn.Module`

- model in this ex.: y = final(nonlinear(linear(BoW)))

  - input: BoW (bag of words vector)
    - usually high dimensional, one slot per word in the vocabulary

  - linear layer: `nn.Linear` transforms the BoW into a smaller hidden representation (weighted sum + bias)
    
  - nonlinear activation: `nn.ReLU` applies a nonlinear function - so the model can capture more complex patterns

  - final linear layer: `nn.Linear` maps the hidden representation to output scores (e.g. one per class)



``` python
#5. define custom model - that pytorch recognizes as a proper neural network 

class BoWClassifier(nn.Module): #using the BoW classifier

    def __init__(self, num_labels, vocab_size, num_hidden = 2): w
        # Calls the init function of nn.Module. 
        super(BoWClassifier, self).__init__()
    '''
    - define an object from the class (_init_ initializes/sets up the object's attributes)
    - self: defining this allows for accessing and storing data within that object
    - num_labels: predefined # of labels
    - vocab_size: # of vocab words - which determines size of Bag of Words vector
    - num_hidden = 2: specifies the default number of hidden units/layers of 2
    - super () initializes/sets up the nn.Module parent class 
    '''

        # Define the parameters that you need.
        self.linear = nn.Linear(vocab_size, num_hidden)
        '''
        layer 1: linear
        y = Wx + b
        - W = weight matrix (how important each input feature is)
        - x = input vector (e.g. BoW vector)
        - b = bias vector
        y = output 

        - input shape (x): vocab_size (BoW vector size)
        - output shape (y): # hidden layers 

        --> learns weights and bias here
        '''

        # non-linearity (here it is also a layer!)
        self.nonlinear = nn.ReLU()
        '''
        layer 2: nonlinear 
        - applies the ReLU activation function
        - each node finds its own weight via gradient descent loss function 
        '''

        # final affine transformation
        self.final = nn.Linear(num_hidden, num_labels)
        '''
        output layer: linear

        - input shape: num_hidden (# hidden layers)
        - output shape: num_labels (# of labels)
        --> takes numbers from the hidden layers, turns into output numbers (1 per class)
        '''

    def forward(self, bow_vec): 
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return self.final(self.nonlinear(self.linear(bow_vec)))
    '''
    define computation (how input data flows thru the model) 
    - forward (self, x)
        - x is the data that the model is supposed to process
        - for text model, can be BoW (this ex.), embeddings, etc. 

    - self.linear layer: (defined above) takes BoW vector (as a tensor, so it can run on GPU) as input, produces numbers for the hidden layers (of num_hidden shape)
    - self.nonlinear: (defined above) applies nonlinear ReLU activation function
    - self.final: (defined above) turns hidden features to output class scores

    '''
```
softmax comes in during training phase - so the output numbers get turned into probabilities for each class

# 3. create BoW vector via 2 functions
``` python

#6. define function to create BoW vector 

def make_bow_vector(sentence, word_to_ix): #inputs are sentence (list of words), and dictionary with key: word | value: index
    vec = torch.zeros(len(word_to_ix)) #create an empty vector (full of zeroes), with length = size of vocab list (so each slot corresponds to a word in the vocab)
    for word in sentence: #for each word in the sentence
        vec[word_to_ix[word]] += 1 #looks up the index of the word in the dict - then access that slot position in vec - then increase that slot by 1 (to count one occurrence of this word, bc starting at 0)
    return vec.view(1, -1) #shapes the vector, so get 1 sentence, with inferred vocab size -> e.g. 2D tensor [[1,1,1,0]]

#7. convert label into a numeric tensor the model can learn from 
def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

'''
inputs: 
- label: class (e.g. Spanish or English)
- label_to_ix: dictionary mapping each label to an integer 

output:
- a tensor containing the numeric index of the label 
'''

```
# 4. create classifier (model) to store model's parameters 

``` python

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE) #recall that these inputs are predefined 

# to print params:
for param in model.parameters():
    print(param)

```

# 5. Before training - running the model on test data to compare with the results from a trained model 

``` python

    with torch.no_grad(): #no gradient descent; makes it faster and saves memory
    for text, label in test_data: #for each pair (text, then label of what language it is)
        bow_vec = make_bow_vector(text, word_to_ix) #create a BoW tensor using the pre-defined function, with inputs text (sentence) and dictionary with key: word | value: index
        log_probs = model(bow_vec) #use our defined model, with input the BoW tensor - output the raw class scores (before they are turned into probabilities)
        print(log_probs) #since the model is untrained, these numbers (one per input sentence) are random

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])

'''
looks inside the model's weight (parameter) matrix to see how it treats the word 'creo'
- next(model.paramters()) runs through specifically the weight matrix of the 1st linear layer
- gets the index of word 'creo' from the word_to_ix dict 
- selects the column of the weight matrix that belongs to 'creo' 
'''
```

# 4. training
We set our loss function to cross entropy which combines `nn.LogSoftmax()` and `nn.NLLLoss()` (negative log likelihood) and calculate gradients with stochastic gradient descent.

Usually we want to pass over the training data several times by setting a respective number of epochs. Since we have a tiny dataset (in this example), we will exaggerate with the number of epochs.

``` python

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200): 
    for instance, label in data:
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Tensor as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = make_bow_vector(instance, word_to_ix) #create BoW tensor (using the predefined function), inputting the sentence and the dictionary - so that can fill out the tensor with a word and its index 
        target = make_target(label, label_to_ix) #create a target (using the predefined function), inputting label and the dictionary - so have a tensor target

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

```

# 5. evaluation
Let's see if our model can now predict more accurately, if a sentence is written in English or Spanish! 

Indeed, the log probability for Spanish is much higher for the first sentence, while the log probability for English is much higher for the second sentence in the test data!

``` python

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

print(next(model.parameters())[:, word_to_ix["creo"]])

```