# -*- coding: utf-8 -*-


from tkinter import *
from tkinter import messagebox
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


df=pd.read_csv("spam.csv",encoding='ISO-8859-1')
df.drop('Unnamed: 2',axis=1,inplace=True)
df.drop('Unnamed: 3',axis=1,inplace=True)
df.drop('Unnamed: 4',axis=1,inplace=True)


# Randomize the dataset
data_randomized = df.sample(frac=1, random_state=1)
# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)
data=training_set.to_numpy()
spam_messages=data[:, 1] # array where each element is a spam message
classification= data[:, 0] # array where each element is either 'ham' or 'spam'




tokenizer=RegexpTokenizer('\w+') # used to break sentences into words
sw=set(stopwords.words('english')) # set containing stopwords of english language is created
ps=PorterStemmer() # used to replace certain words with their simpler forms eg: 'running' becomes 'run'(meaning is not changed)


#we have to perform tokenization, removal of stopwords and stemming operatins on each of the spam messages present in array spam_messages
def operation(spam_messages):
    operated_messages=[]
    for spam in spam_messages:  #  for each spam message
        spam=spam.lower() # convert letters to lowercase
        tokenized_spam=tokenizer.tokenize(spam) # sentence is broken down into words(tokenization). tokenized_spam is a list of separated words
        sw_removed_spam=[word for word in tokenized_spam if word not in sw] # removing the stopwords from tokenized_spam list
        stemmed_spam=[ps.stem(word) for word in sw_removed_spam] #stemming the words present in sw_removed_spam list
        clean_message=' '.join(stemmed_spam) #the words of stemmed_spam list are joined to create a string. This string has been obtained after applying the 3 operations listed above.
        operated_messages.append(clean_message) #the cleaned spam message is added to this list
    return operated_messages
 #-----------function end----------------------------------------------------------------------       

operated_messages=operation(spam_messages) #list containing clean spam messages is obtained
#creating vocabulary
training_set['v2']=operated_messages #putting the cleaned message in 'v2' column of training_set
training_set['v2']=training_set['v2'].str.split() #creating word lists in each row of 'v2' column
vocabulary=[] #vocabulary will contain all the unique words across the training_set's spam messages
for words_list in training_set['v2']:
    for word in words_list:
        vocabulary.append(word)
vocabulary=list(set(vocabulary)) #6484 unique words across all cleaned messages

#Creating another dataframe by making use of a dictionary
word_counts_per_sms = {unique_word: [0] * len(training_set['v2']) for unique_word in vocabulary}

for index, words_list in enumerate(training_set['v2']):
    for word in words_list:
        word_counts_per_sms[word][index] += 1
word_counts = pd.DataFrame(word_counts_per_sms)
print(word_counts.head())

training_set_clean = pd.concat([training_set, word_counts], axis=1)#combining the training_set and word_counts
print(training_set_clean.head())                                    # dataframes to a single dataframe

#eg:- training_set_clean['convey'][0]
# training_set_clean['regard'][0]
# training_set_clean['u'][1]
# training_set_clean['even'][1]


#naive bayes classifier implementation---------------------------------------------------------------------------------

# Isolating spam and ham messages first(dataframe split into two dataframes)
spam_messages = training_set_clean[training_set_clean['v1'] == 'spam']#spam dataframe
ham_messages = training_set_clean[training_set_clean['v1'] == 'ham']#ham dataframe

# P(Spam) and P(Ham) calculation (constant)
p_spam = len(spam_messages) / len(training_set_clean) #no of spam messages divided by total no of messages
p_ham = len(ham_messages) / len(training_set_clean) #no of ham messages divided by total no of messages

# N_Spam calculation (constant)
n_words_per_spam_message = spam_messages['v2'].apply(len)#gives a series(1 column)
n_spam = n_words_per_spam_message.sum() #total no of words in spam messages

# N_Ham calculation (constant)
n_words_per_ham_message = ham_messages['v2'].apply(len)
n_ham = n_words_per_ham_message.sum() #total no of words in ham messages

# N_Vocabulary calculation (constant)
n_vocabulary = len(vocabulary) #total no of words in vocabulary

# Laplace smoothing parameter (constant)
alpha = 1 

# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary} #probability that given word is spam 
parameters_ham = {unique_word:0 for unique_word in vocabulary}  #probability that given word is ham

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham



def classify(message):  #Naive bayes classifier function
    mylist=[message] #converting to list as operation function requires list or array as input
    mylist=operation(mylist) #cleaning the message
    message=mylist[0] #cleaned message is obtained
    message=message.split() #Converting to list where each element of the list is a word of the sentence
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
    
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'

#classification of test_set
test_set['predicted'] = test_set['v2'].apply(classify) #adds another column to test set
print(test_set.head())
#checking accuracy of classifier 
correct = 0
total = test_set.shape[0]

for row in test_set.iterrows(): #iterrows() returns (index,series) pairs 
    row=row[1] #we want the second element of the tuple(series) not the index
    if row['v1'] == row['predicted']:
      correct += 1

print("\n")
print('Total no of messages in test_set:',total)
print('Correctly Classified:', correct)
print('Incorrectly Classified:', total - correct)
print('Accuracy of classifier:', correct/total)

#confusion matrix
from sklearn.metrics import confusion_matrix
y_actual=test_set['v1'].to_numpy()
y_predicted=test_set['predicted'].to_numpy()
cm=confusion_matrix(y_actual,y_predicted)
print("Confusion matrix:-")
print(cm) #1st row-ham,2nd row-spam,1st column-ham,2nd column-spam
'''out of 972+1=973 ham messages in test_set, the classifier classified 972 messages correctly as ham
and classifed 1 ham message incorrectly as spam.
out of 4+137=141 spam messages in test_set, the classifier classified 137 messages correctly 
as spam and classified 4 spam messages incorrectly as ham '''



#------------------------------------------------GUI---------------------------------------------------------------------------
def myClick():
    message=myTextBox.get(1.0,END)
    result=classify(message)
    if(result=='ham'):
        messagebox.showinfo("Result","The entered message is not a spam message")
    elif(result=='spam'):
        messagebox.showinfo("Result","The entered message is a spam message!")
    else:
        messagebox.showerror("ERROR!","The entered message cannot be classified to either group!")
    myTextBox.delete(1.0,END)
        
root=Tk()
root.title("Spam message checker")
root.iconbitmap('spam.ico')
myLabel=Label(root,text="Enter the message:")
myTextBox=Text(root,width=60,height=20)
myButton=Button(root,text="Check",command=myClick)
myLabel.grid(row=0,column=0)
myTextBox.grid(row=1,column=0)
myButton.grid(row=2,column=0)
root.mainloop()

