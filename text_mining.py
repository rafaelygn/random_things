# Import Lib
import re
from nltk import FreqDist

# set the path
path = 'C:\\Users\\Rafael\\Desktop\\Data Science\\NLP\\'

# Creating a 'big string' with all words together
text_full = ''
f = open(path+'TF.txt', 'r')
text = f.readlines()
for line in text:
    text_full += line

# Tokenization
token_txt_v0 = nltk.word_tokenize(text_full)
token_txt = []
for word in token_txt_v0:
    token_txt.append(word.lower())

# Vocab Analysis
print(f'Nº de palavras: {len(token_txt_v0)} \n',
      f'Vocabulário: {len(set(token_txt_v0))}')

print(f'Nº de palavras: {len(token_txt)} \n',
      f'Vocabulário: {len(set(token_txt))}')

# Most Common Strings
fd = FreqDist(token_txt)

# How many for a specific word
fd[';']
fd['que']
fd['Figura']
fd['figura']

# Plot Cumulative
fd.plot(50, cumulative=True)

# Using Regex
exampleString = '''
Jessica is 15 old, and Daniel is 27 years old.
Edward is 97, and his 102 grandfather, Oscar is 102'''

re.findall(r'\d{3,4}', exampleString)
set(re.findall(r'[Ff]iguras?', text_full))
