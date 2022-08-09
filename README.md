# Word Vectorization - Lab

## Introduction

In this lab, you'll tokenize and vectorize text documents, create and use a bag of words, and identify words unique to individual documents using TF-IDF vectorization. 


## Objectives

In this lab you will:  

- Implement tokenization and count vectorization from scratch 
- Implement TF-IDF from scratch 
- Use dimensionality reduction on vectorized text data to create and interpret visualizations 

## Let's get started!

Run the cell below to import everything necessary for this lab.  


```python
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
np.random.seed(0)
```


```python
# __SOLUTION__ 
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
np.random.seed(0)
```

### Our Corpus

In this lab, we'll be working with 20 different documents, each containing song lyrics from either Garth Brooks or Kendrick Lamar albums.  

The songs are contained within the `data` subdirectory, contained within the same folder as this lab.  Each song is stored in a single file, with files ranging from `song1.txt` to `song20.txt`.  

To make it easy to read in all of the documents, use a list comprehension to create a list containing the name of every single song file in the cell below. 


```python
filenames = None
```


```python
# __SOLUTION__ 
filenames = ['song' + str(i) + '.txt' for i in range(1, 21)]
```

Next, create an empty DataFrame called `songs_df`.  As we read in the songs and store and clean them, we'll store them in this DataFrame.


```python
songs_df = None
```


```python
# __SOLUTION__ 
songs_df = pd.DataFrame()
```

Next, let's import a single song to see what our text looks like so that we can make sure we clean and tokenize it correctly. 

In the cell below, read in and print out the lyrics from `song11.txt`.  Use vanilla Python, no pandas needed.  


```python
# Import and print song11.txt

```


```python
# __SOLUTION__ 
with open('data/song11.txt') as f:
    test_song = f.readlines()
    print(test_song)
```

    ['[Kendrick Lamar:]\n', "Love, let's talk about love\n", 'Is it anything and everything you hoped for?\n', 'Or do the feeling haunt you?\n', 'I know the feeling haunt you\n', '[SZA:]\n', 'This may be the night that my dreams might let me know\n', 'All the stars approach you, all the stars approach you, all the stars approach you\n', 'This may be the night that my dreams might let me know\n', 'All the stars are closer, all the stars are closer, all the stars are closer\n', '[Kendrick Lamar:]\n', "Tell me what you gon' do to me\n", "Confrontation ain't nothin' new to me\n", 'You can bring a bullet, bring a sword, bring a morgue\n', "But you can't bring the truth to me\n", 'Fuck you and all your expectations\n', "I don't even want your congratulations\n", 'I recognize your false confidence\n', 'And calculated promises all in your conversation\n', 'I hate people that feel entitled\n', "Look at me crazy 'cause I ain't invite you\n", 'Oh, you important?\n', "You the moral to the story? You endorsin'?\n", "Motherfucker, I don't even like you\n", "Corrupt a man's heart with a gift\n", "That's how you find out who you dealin' with\n", "A small percentage who I'm buildin' with\n", "I want the credit if I'm losin' or I'm winnin'\n", "On my momma, that's the realest shit\n", "Love, let's talk about love\n", 'Is it anything and everything you hoped for?\n', 'Or do the feeling haunt you?\n', 'I know the feeling haunt you\n', '[SZA:]\n', 'This may be the night that my dreams might let me know\n', 'All the stars approach you, all the stars approach you, all the stars approach you\n', 'This may be the night that my dreams might let me know\n', 'All the stars are closer, all the stars are closer, all the stars are closer\n', 'Skin covered in ego\n', "Get to talkin' like ya involved, like a rebound\n", 'Got no end game, got no reason\n', "Got to stay down, it's the way that you making me feel\n", 'Like nobody ever loved me like you do, you do\n', "You kinda feeling like you're tryna get away from me\n", "If you do, I won't move\n", "I ain't just cryin' for no reason\n", "I ain't just prayin' for no reason\n", 'I give thanks for the days, for the hours\n', "And another way, another life breathin'\n", "I did it all 'cause it feel good\n", "I wouldn't do it at all if it feel bad\n", "Better live your life, we're runnin' out of time\n", '[Kendrick Lamar & SZA:]\n', "Love, let's talk about love\n", 'Is it anything and everything you hoped for?\n', 'Or do the feeling haunt you?\n', 'I know the feeling haunt you\n', '[SZA:]\n', 'This may be the night that my dreams might let me know\n', 'All the stars approach you, all the stars approach you, all the stars approach you\n', 'This may be the night that my dreams might let me know\n', 'All the stars are closer, all the stars are closer, all the stars are closer\n']


### Tokenizing our Data

Before we can create a bag of words or vectorize each document, we need to clean it up and split each song into an array of individual words. Computers are very particular about strings. If we tokenized our data in its current state, we would run into the following problems:

- Counting things that aren't actually words.  In the example above, `"[Kendrick]"` is a note specifying who is speaking, not a lyric contained in the actual song, so it should be removed.  
- Punctuation and capitalization would mess up our word counts. To the Python interpreter, `love`, `Love`, `Love?`, and `Love\n` are all unique words, and would all be counted separately.  We need to remove punctuation and capitalization, so that all words will be counted correctly. 

Consider the following sentences from the example above:

`"Love, let's talk about love\n", 'Is it anything and everything you hoped for?\n'`

After tokenization, this should look like:

`['love', 'let's', 'talk', 'about', 'love', 'is', 'it', 'anything', 'and', 'everything', 'you', 'hoped', 'for']`

Tokenization is pretty tedious if we handle it manually, and would probably make use of regular expressions, which is outside the scope of this lab. In order to keep this lab moving, we'll use a library function to clean and tokenize our data so that we can move onto vectorization.  

Tokenization is a required task for just about any Natural Language Processing (NLP) task, so great industry-standard tools exist to tokenize things for us, so that we can spend our time on more important tasks without getting bogged down hunting every special symbol or punctuation in a massive dataset. For this lab, we'll make use of the tokenizer in the amazing `nltk` library, which is short for _Natural Language Tool Kit_.

**_NOTE:_** NLTK requires extra installation methods to be run the first time certain methods are used.  If `nltk` throws you an error about needing to install additional packages, follow the instructions in the error message to install the dependencies, and then rerun the cell.  

Before we tokenize our songs, we'll do only a small manual bit of cleaning. In the cell below, write a function that allows us to remove lines that have `['artist names']` in it, to ensure that our song files contain only lyrics that are actually in the song. For the lines that remain, make every word lowercase, remove newline characters `\n`, and all the following punctuation marks: `",.'?!"`

Test the function on `test_song` to show that it has successfully removed `'[Kendrick Lamar:]'` and other instances of artist names from the song and returned it.  


```python
def clean_song(song):
    pass

song_without_brackets = None
print(song_without_brackets)
```


```python
# __SOLUTION__ 
def clean_song(song):
    cleaned_song = []
    for line in song:
        if not '[' in line and  not ']' in line:
            for symbol in ",.?!''\n":
                line = line.replace(symbol, '').lower()
            cleaned_song.append(line)

    return cleaned_song

song_without_brackets = clean_song(test_song)
song_without_brackets
```




    ['love lets talk about love',
     'is it anything and everything you hoped for',
     'or do the feeling haunt you',
     'i know the feeling haunt you',
     'this may be the night that my dreams might let me know',
     'all the stars approach you all the stars approach you all the stars approach you',
     'this may be the night that my dreams might let me know',
     'all the stars are closer all the stars are closer all the stars are closer',
     'tell me what you gon do to me',
     'confrontation aint nothin new to me',
     'you can bring a bullet bring a sword bring a morgue',
     'but you cant bring the truth to me',
     'fuck you and all your expectations',
     'i dont even want your congratulations',
     'i recognize your false confidence',
     'and calculated promises all in your conversation',
     'i hate people that feel entitled',
     'look at me crazy cause i aint invite you',
     'oh you important',
     'you the moral to the story you endorsin',
     'motherfucker i dont even like you',
     'corrupt a mans heart with a gift',
     'thats how you find out who you dealin with',
     'a small percentage who im buildin with',
     'i want the credit if im losin or im winnin',
     'on my momma thats the realest shit',
     'love lets talk about love',
     'is it anything and everything you hoped for',
     'or do the feeling haunt you',
     'i know the feeling haunt you',
     'this may be the night that my dreams might let me know',
     'all the stars approach you all the stars approach you all the stars approach you',
     'this may be the night that my dreams might let me know',
     'all the stars are closer all the stars are closer all the stars are closer',
     'skin covered in ego',
     'get to talkin like ya involved like a rebound',
     'got no end game got no reason',
     'got to stay down its the way that you making me feel',
     'like nobody ever loved me like you do you do',
     'you kinda feeling like youre tryna get away from me',
     'if you do i wont move',
     'i aint just cryin for no reason',
     'i aint just prayin for no reason',
     'i give thanks for the days for the hours',
     'and another way another life breathin',
     'i did it all cause it feel good',
     'i wouldnt do it at all if it feel bad',
     'better live your life were runnin out of time',
     'love lets talk about love',
     'is it anything and everything you hoped for',
     'or do the feeling haunt you',
     'i know the feeling haunt you',
     'this may be the night that my dreams might let me know',
     'all the stars approach you all the stars approach you all the stars approach you',
     'this may be the night that my dreams might let me know',
     'all the stars are closer all the stars are closer all the stars are closer']



Great. Now, write a function `tokenize()` that takes in songs that have had their brackets removed, joins all of the lines into a single string, and then uses `word_tokenize()` on it to get a fully tokenized version of the song.  Test this function on `song_without_brackets` to ensure that the function works. 


```python
def tokenize(song):
    pass

tokenized_test_song = None
tokenized_test_song[:10]
```


```python
# __SOLUTION__ 
def tokenize(song):
    joined_song = ' '.join(song)
    tokenized_song = word_tokenize(joined_song)
    
    return tokenized_song

tokenized_test_song = tokenize(song_without_brackets)
tokenized_test_song[:10]
```




    ['love',
     'lets',
     'talk',
     'about',
     'love',
     'is',
     'it',
     'anything',
     'and',
     'everything']



Great! Now that we can tokenize our songs, we can move onto vectorization. 


### Count Vectorization

Machine Learning algorithms don't understand strings. However, they do understand math, which means they understand vectors and matrices.  By **_Vectorizing_** the text, we just convert the entire text into a vector, where each element in the vector represents a different word. The vector is the length of the entire vocabulary -- usually, every word that occurs in the English language, or at least every word that appears in our corpus.  Any given sentence can then be represented as a vector where all the vector is 1 (or some other value) for each time that word appears in the sentence. 

Consider the following example: 

<center>"I scream, you scream, we all scream for ice cream."</center>

| 'aardvark' | 'apple' | [...] | 'I' | 'you' | 'scream' | 'we' | 'all' | 'for' | 'ice' | 'cream' | [...] | 'xylophone' | 'zebra' |
|:----------:|:-------:|:-----:|:---:|:-----:|:--------:|:----:|:-----:|:-----:|:-----:|:-------:|:-----:|:-----------:|:-------:|
|      0     |    0    |   0   |  1  |   1   |     3    |   1  |   1   |   1   |   1   |    1    |   0   |      0      |    0    |

This is called a **_Sparse Representation_**, since the strong majority of the columns will have a value of 0.  Note that elements corresponding to words that do not occur in the sentence have a value of 0, while words that do appear in the sentence have a value of 1 (or 1 for each time it appears in the sentence).

Alternatively, we can represent this sentence as a plain old Python dictionary of word frequency counts:

```python
BoW = {
    'I':1,
    'you':1,
    'scream':3,
    'we':1,
    'all':1,
    'for':1,
    'ice':1,
    'cream':1
}
```

Both of these are examples of **_Count Vectorization_**. They allow us to represent a sentence as a vector, with each element in the vector corresponding to how many times that word is used.

#### Positional Information and Bag of Words

Notice that when we vectorize a sentence this way, we lose the order that the words were in.  This is the **_Bag of Words_** approach mentioned earlier.  Note that sentences that contain the same words will create the same vectors, even if they mean different things -- e.g. `'cats are scared of dogs'` and `'dogs are scared of cats'` would both produce the exact same vector, since they contain the same words.  

In the cell below, create a function that takes in a tokenized, cleaned song and returns a count vectorized representation of it as a Python dictionary. Add in an optional parameter called `vocab` that defaults to `None`. This way, if we are using a vocabulary that contains words not seen in the song, we can still use this function by passing it into the `vocab` parameter. 

**_Hint:_**  Consider using a `set()` to make this easier!


```python
def count_vectorize(song, vocab=None):
    pass

test_vectorized = None
print(test_vectorized)
```


```python
# __SOLUTION__ 
def count_vectorize(song, vocab=None):
    if vocab:
        unique_words = vocab
    else:
        unique_words = list(set(song))
    
    song_dict = {i:0 for i in unique_words}
    
    for word in song:
        song_dict[word] += 1
    
    return song_dict

test_vectorized = count_vectorize(tokenized_test_song)
print(test_vectorized)
```

    {'conversation': 1, 'to': 6, 'know': 9, 'days': 1, 'down': 1, 'story': 1, 'im': 3, 'good': 1, 'even': 2, 'stars': 18, 'dont': 2, 'stay': 1, 'youre': 1, 'making': 1, 'feeling': 7, 'nobody': 1, 'night': 6, 'congratulations': 1, 'me': 14, 'all': 22, 'ever': 1, 'endorsin': 1, 'credit': 1, 'look': 1, 'thanks': 1, 'tell': 1, 'nothin': 1, 'find': 1, 'you': 34, 'on': 1, 'were': 1, 'bullet': 1, 'important': 1, 'haunt': 6, 'confrontation': 1, 'crazy': 1, 'gift': 1, 'get': 2, 'let': 6, 'time': 1, 'false': 1, 'in': 2, 'love': 6, 'losin': 1, 'new': 1, 'realest': 1, 'confidence': 1, 'be': 6, 'entitled': 1, 'corrupt': 1, 'are': 9, 'dreams': 6, 'closer': 9, 'at': 2, 'reason': 3, 'what': 1, 'gon': 1, 'tryna': 1, 'people': 1, 'mans': 1, 'sword': 1, 'life': 2, 'thats': 2, 'just': 2, 'want': 2, 'prayin': 1, 'how': 1, 'better': 1, 'dealin': 1, 'with': 3, 'it': 7, 'kinda': 1, 'give': 1, 'end': 1, 'small': 1, 'from': 1, 'percentage': 1, 'bring': 4, 'can': 1, 'might': 6, 'live': 1, 'shit': 1, 'did': 1, 'game': 1, 'its': 1, 'way': 2, 'that': 8, 'ego': 1, 'this': 6, 'but': 1, 'truth': 1, 'rebound': 1, 'promises': 1, 'or': 4, 'loved': 1, 'motherfucker': 1, 'hate': 1, 'oh': 1, 'the': 38, 'covered': 1, 'cryin': 1, 'aint': 4, 'buildin': 1, 'do': 8, 'invite': 1, 'for': 7, 'your': 5, 'is': 3, 'calculated': 1, 'fuck': 1, 'like': 6, 'breathin': 1, 'lets': 3, 'talk': 3, 'my': 7, 'if': 3, 'runnin': 1, 'feel': 4, 'of': 1, 'expectations': 1, 'talkin': 1, 'ya': 1, 'hours': 1, 'a': 7, 'another': 2, 'cant': 1, 'got': 3, 'about': 3, 'moral': 1, 'out': 2, 'momma': 1, 'winnin': 1, 'morgue': 1, 'involved': 1, 'i': 15, 'recognize': 1, 'and': 6, 'cause': 2, 'everything': 3, 'approach': 9, 'away': 1, 'wont': 1, 'bad': 1, 'who': 2, 'skin': 1, 'move': 1, 'may': 6, 'anything': 3, 'no': 4, 'hoped': 3, 'wouldnt': 1, 'heart': 1}


Great! You've just successfully vectorized your first text document! Now, let's look at a more advanced type of vectorization, TF-IDF!

### TF-IDF Vectorization

TF-IDF stands for **_Term Frequency, Inverse Document Frequency_**.  This is a more advanced form of vectorization that weighs each term in a document by how unique it is to the given document it is contained in, which allows us to summarize the contents of a document using a few key words.  If the word is used often in many other documents, it is not unique, and therefore probably not too useful if we wanted to figure out how this document is unique in relation to other documents. Conversely, if a word is used many times in a document, but rarely in all the other documents we are considering, then it is likely a good indicator for telling us that this word is important to the document in question.  

The formula TF-IDF uses to determine the weights of each term in a document is **_Term Frequency_** multiplied by **_Inverse Document Frequency_**, where the formula for Term Frequency is:

$$\large Term\ Frequency(t) = \frac{number\ of\ times\ t\ appears\ in\ a\ document} {total\ number\ of\ terms\ in\ the\ document} $$
<br>
<br>
Complete the following function below to calculate term frequency for every term in a document.  


```python
def term_frequency(BoW_dict):
    pass

test = None
print(list(test)[10:20])
```


```python
# __SOLUTION__ 
def term_frequency(BoW_dict):
    total_word_count = sum(BoW_dict.values())
    
    for ind, val in BoW_dict.items():
        BoW_dict[ind] = val/ total_word_count
    
    return BoW_dict

test = term_frequency(test_vectorized)
print(list(test)[10:20])
```

    ['dont', 'stay', 'youre', 'making', 'feeling', 'nobody', 'night', 'congratulations', 'me', 'all']


Now that we have this, we can easily calculate _Inverse Document Frequency_.  In the cell below, complete the following function.  this function should take in the list of dictionaries, with each item in the list being a bag of words representing the words in a different song. The function should return a dictionary containing the inverse document frequency values for each word.  

The formula for Inverse Document Frequency is:  
<br>  
<br>
$$\large  IDF(t) =  log_e(\frac{Total\ Number\ of\ Documents}{Number\ of\ Documents\ with\ t\ in\ it})$$


```python
def inverse_document_frequency(list_of_dicts):
    pass
```


```python
# __SOLUTION__ 
def inverse_document_frequency(list_of_dicts):
    vocab_set = set()
    # Iterate through list of dfs and add index to vocab_set
    for d in list_of_dicts:
        for word in d.keys():
            vocab_set.add(word)
    
    # Once vocab set is complete, create an empty dictionary with a key for each word and value of 0.
    full_vocab_dict = {i:0 for i in vocab_set}
    
    # Loop through each word in full_vocab_dict
    for word, val in full_vocab_dict.items():
        docs = 0
        
        # Loop through list of dicts.  Each time a dictionary contains the word, increment docs by 1
        for d in list_of_dicts:
            if word in d:
                docs += 1
        
        # Now that we know denominator for equation, compute and set IDF value for word
        
        full_vocab_dict[word] = np.log((len(list_of_dicts)/ float(docs)))
    
    return full_vocab_dict


```

### Computing TF-IDF

Now that we can compute both Term Frequency and Inverse Document Frequency, computing an overall TF-IDF value is simple! All we need to do is multiply the two values.  

In the cell below, complete the `tf_idf()` function.  This function should take in a list of dictionaries, just as the `inverse_document_frequency()` function did.  This function returns a new list of dictionaries, with each dictionary containing the tf-idf vectorized representation of a corresponding song document. 

**_NOTE:_** Each document should contain the full vocabulary of the entire combined corpus.  


```python
def tf_idf(list_of_dicts):
    pass
```


```python
# __SOLUTION__ 
def tf_idf(list_of_dicts):
    # Create empty dictionary containing full vocabulary of entire corpus
    doc_tf_idf = {}
    idf = inverse_document_frequency(list_of_dicts)
    full_vocab_list = {i:0 for i in list(idf.keys())}
    
    # Create tf-idf list of dictionaries, containing a dictionary that will be updated for each document
    tf_idf_list_of_dicts = []
    
    # Now, compute tf and then use this to compute and set tf-idf values for each document
    for doc in list_of_dicts:
        doc_tf = term_frequency(doc)
        for word in doc_tf:
            doc_tf_idf[word] = doc_tf[word] * idf[word]
        tf_idf_list_of_dicts.append(doc_tf_idf)
    
    return tf_idf_list_of_dicts
```

### Vectorizing All Documents

Now that we've created all the necessary helper functions, we can load in all of our documents and run each through the vectorization pipeline we've just created.

In the cell below, complete the `main()` function.  This function should take in a list of file names (provided for you in the `filenames` list we created at the start), and then:

- Read in each document
- Tokenize each document
- Convert each document to a bag of words (dictionary representation)
- Return a list of dictionaries vectorized using tf-idf, where each dictionary is a vectorized representation of a document 


```python
def main(filenames):
    pass

tf_idf_all_docs = None
print(list(tf_idf_all_docs[0])[:10])
```


```python
# __SOLUTION__ 
def main(filenames):
    # Iterate through list of filenames and read each in
    count_vectorized_all_documents = []
    for file in filenames:
        with open('data/' + file) as f:
            raw_data = f.readlines()
        # Clean and tokenize raw text
        cleaned = clean_song(raw_data)
        tokenized = tokenize(cleaned)
        
        # Get count vectorized representation and store in count_vectorized_all_documents  
        count_vectorized_document = count_vectorize(tokenized)
        count_vectorized_all_documents.append(count_vectorized_document)
    
    # Now that we have a list of BoW respresentations of each song, create a tf-idf representation of everything
    tf_idf_all_docs = tf_idf(count_vectorized_all_documents)
    
    return tf_idf_all_docs

tf_idf_all_docs = main(filenames)
print(list(tf_idf_all_docs[0])[:10])
```

    ['raised', 'runneth', '9', 'to', 'jc', 'know', 'gates', 'passed', 'borderline', 'letters']


### Visualizing our Vectorizations

Now that we have a tf-idf representation of each document, we can move on to the fun part -- visualizing everything!

In the cell below, examine our dataset to figure out how many dimensions our dataset has. 

**_HINT_**: Remember that every word is its own dimension!


```python
num_dims = None
print("Number of Dimensions: {}".format(num_dims))
```


```python
# __SOLUTION__ 
num_dims = len(tf_idf_all_docs[0])
print("Number of Dimensions: {}".format(num_dims))
```

    Number of Dimensions: 1344


There are too many dimensions for us to visualize! In order to make it understandable to human eyes, we'll need to reduce it to 2 or 3 dimensions.  

To do this, we'll use a technique called **_t-SNE_** (short for _t-Stochastic Neighbors Embedding_).  This is too complex for us to code ourselves, so we'll make use of scikit-learn's implementation of it.  

First, we need to pull the words out of the dictionaries stored in `tf_idf_all_docs` so that only the values remain, and store them in lists instead of dictionaries.  This is because the t-SNE only works with array-like objects, not dictionaries.  

In the cell below, create a list of lists that contains a list representation of the values of each of the dictionaries stored in `tf_idf_all_docs`.  The same structure should remain -- e.g. the first list should contain only the values that were in the first dictionary in `tf_idf_all_docs`, and so on. 


```python
tf_idf_vals_list = []

for i in tf_idf_all_docs:
    tf_idf_vals_list.append(list(i.values()))
    
tf_idf_vals_list[0][:10]
```


```python
# __SOLUTION__ 
tf_idf_vals_list = []

for i in tf_idf_all_docs:
    tf_idf_vals_list.append(list(i.values()))
    
tf_idf_vals_list[0][:10]
```




    [0.009133330102298753,
     0.027399990306896257,
     0.0035369970706513763,
     0.00012855462252518916,
     0.009133330102298753,
     0.0017878443305199617,
     0.027399990306896257,
     0.009133330102298753,
     0.009133330102298753,
     0.009133330102298753]



Now that we have only the values, we can use the `TSNE()` class from `sklearn` to transform our data appropriately.  In the cell below, instantiate `TSNE()` with `n_components=3`.  Then, use the created object's `.fit_transform()` method to transform the data stored in `tf_idf_vals_list` into 3-dimensional data.  Then, inspect the newly transformed data to confirm that it has the correct dimensionality. 


```python
t_sne_object_3d = None
transformed_data_3d = None
transformed_data_3d
```


```python
# __SOLUTION__ 
t_sne_object_3d = TSNE(n_components=3)
transformed_data_3d = t_sne_object_3d.fit_transform(tf_idf_vals_list)
transformed_data_3d
```




    array([[ -85.04439  ,  -12.42539  ,  160.53555  ],
           [  -3.4193013, -217.31935  ,   23.678131 ],
           [-181.40318  ,   37.18949  , -279.00134  ],
           [-168.894    ,  165.66945  ,   74.69088  ],
           [ 174.79863  ,  -41.690086 ,  301.93756  ],
           [  38.269165 ,  151.16583  ,   93.36936  ],
           [  31.328133 , -244.56982  , -167.79346  ],
           [  27.850698 ,   95.67248  ,  277.84744  ],
           [-125.99564  ,   96.77036  , -110.62522  ],
           [ 250.55006  ,  192.88551  , -168.65764  ],
           [-146.47461  , -198.77888  ,  159.63432  ],
           [ 146.62701  ,   29.345417 , -164.25223  ],
           [ 192.62463  , -140.5079   ,  -58.02724  ],
           [ 174.86421  ,  246.83163  ,   14.580853 ],
           [  19.869684 ,  211.12953  , -123.161224 ],
           [-216.70706  ,  -50.75892  ,    5.941306 ],
           [-107.28444  , -112.47715  , -158.13704  ],
           [ 247.58127  ,   34.86669  ,   84.8679   ],
           [  21.53436  ,   -9.539048 ,  -21.645472 ],
           [ 107.66281  , -111.01468  ,  139.19974  ]], dtype=float32)



We'll also want to check out how the visualization looks in 2d.  Repeat the process above, but this time, instantiate `TSNE()` with 2 components instead of 3.  Again, use `.fit_transform()` to transform the data and store it in the variable below, and then inspect it to confirm the transformed data has only 2 dimensions. 


```python
t_sne_object_2d = None
transformed_data_2d = None
transformed_data_2d
```


```python
# __SOLUTION__ 
t_sne_object_2d = TSNE(n_components=2)
transformed_data_2d = t_sne_object_2d.fit_transform(tf_idf_vals_list)
transformed_data_2d
```




    array([[ -57.13      ,  -82.44527   ],
           [-149.95668   , -100.78645   ],
           [  17.649107  ,  127.99021   ],
           [-136.68973   ,   99.35129   ],
           [ 211.75693   ,   81.607994  ],
           [ 197.99387   ,  -44.276165  ],
           [ 135.04065   ,   29.907883  ],
           [  55.82382   ,  212.21283   ],
           [ -67.15371   ,  175.84656   ],
           [ -93.41145   ,   -1.3688639 ],
           [  -0.96264917,  -14.616862  ],
           [ -57.00879   , -177.24257   ],
           [-185.9646    ,    5.108982  ],
           [  87.43467   ,  -41.514145  ],
           [ -43.565823  ,   67.657135  ],
           [  53.59356   , -193.05151   ],
           [  29.162214  , -103.39789   ],
           [  51.815746  ,   49.444122  ],
           [ 118.36375   ,  128.68997   ],
           [ 136.48239   , -126.846825  ]], dtype=float32)



Now, let's visualize everything!  Run the cell below to view both 3D and 2D visualizations of the songs.


```python
kendrick_3d = transformed_data_3d[:10]
k3_x = [i[0] for i in kendrick_3d]
k3_y = [i[1] for i in kendrick_3d]
k3_z = [i[2] for i in kendrick_3d]

garth_3d = transformed_data_3d[10:]
g3_x = [i[0] for i in garth_3d]
g3_y = [i[1] for i in garth_3d]
g3_z = [i[2] for i in garth_3d]

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(k3_x, k3_y, k3_z, c='b', s=60, label='Kendrick')
ax.scatter(g3_x, g3_y, g3_z, c='red', s=60, label='Garth')
ax.view_init(30, 10)
ax.legend()
plt.show()

kendrick_2d = transformed_data_2d[:10]
k2_x = [i[0] for i in kendrick_2d]
k2_y = [i[1] for i in kendrick_2d]

garth_2d = transformed_data_2d[10:]
g2_x = [i[0] for i in garth_2d]
g2_y = [i[1] for i in garth_2d]

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(222)
ax.scatter(k2_x, k2_y, c='b', label='Kendrick')
ax.scatter(g2_x, g2_y, c='red', label='Garth')
ax.legend()
plt.show()
```


```python
# __SOLUTION__ 
kendrick_3d = transformed_data_3d[10:]
k3_x = [i[0] for i in kendrick_3d]
k3_y = [i[1] for i in kendrick_3d]
k3_z = [i[2] for i in kendrick_3d]

garth_3d = transformed_data_3d[:10]
g3_x = [i[0] for i in garth_3d]
g3_y = [i[1] for i in garth_3d]
g3_z = [i[2] for i in garth_3d]

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(k3_x, k3_y, k3_z, c='b', s=60, label='Kendrick')
ax.scatter(g3_x, g3_y, g3_z, c='red', s=60, label='Garth')
ax.view_init(30, 10)
ax.legend()
plt.show()

kendrick_2d = transformed_data_2d[:10]
k2_x = [i[0] for i in kendrick_2d]
k2_y = [i[1] for i in kendrick_2d]

garth_2d = transformed_data_2d[10:]
g2_x = [i[0] for i in garth_2d]
g2_y = [i[1] for i in garth_2d]

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(222)
ax.scatter(k2_x, k2_y, c='b', label='Kendrick')
ax.scatter(g2_x, g2_y, c='red', label='Garth')
ax.legend()
plt.show()
```


    
![png](index_files/index_47_0.png)
    



    
![png](index_files/index_47_1.png)
    


Interesting! Take a crack at interpreting these graphs by answering the following questions below:

What does each graph mean? Do you find one graph more informative than the other? Do you think that this method shows us discernable differences between Kendrick Lamar songs and Garth Brooks songs?  Use the graphs and your understanding of TF-IDF to support your answer.  

Write your answer to this question below this line:
________________________________________________________________________________________________________________________________

Both graphs show a basic trend among the red and blue dots, although the 3-dimensional graph is more informative than the 2-dimensional graph. We see a separation between the two artists because they both have words that they use, but the other artist does not. The words in each song that are common to both are reduced to very small numbers or to 0, because of the log operation in the IDF function.  This means that the elements of each song vector with the highest values will be the ones that have words that are unique to that specific document, or at least are rarely used in others.  

## Summary

In this lab, you learned how to: 
* Tokenize a corpus of words and identify the different choices to be made while parsing them 
* Use a count vectorization strategy to create a bag of words
* Use TF-IDF vectorization with multiple documents to identify words that are important/unique to certain documents
* Visualize and compare vectorized text documents
