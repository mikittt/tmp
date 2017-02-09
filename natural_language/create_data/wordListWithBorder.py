from collections import Counter 
import json
import re
#1文ごとに改行され、分かち書きされているtextファイルからmin_words以上の単語数のもののみを文とみなし、その中でborder以上の出現頻度の単語のみで構成されている文とその単語リストを保存する
#文末に何か目印を入れるときは適宜
#長さ最大max_length+1('EOS')の文が入ったリストを返す
f=open('talk_data/train_data.txt')
line=f.readline()

#文の最低単語数
min_words=5
max_length=23

sen=[]
words=[]
while line:
    line=re.sub(r'\n','',line)
    wordList=line.split(' ')
    sentence=[]
    k=1
    for word in wordList:
        if word is not '':
            sentence.append(word)
            k=k+1
    sentence.append('EOS')
    if k>min_words and k<max_length+1:
        sen.append(sentence)
        for part in sentence:
            words.append(part)
    line=f.readline()
f.close        

#最低出現頻度
border=5

a=Counter(words)

wordList=[]
for word in a:
    if a[word]>border-1:
        wordList.append(word)

sentenceList=[]

for sentence in sen:
    eachSentence=[]
    for word in sentence:
        if word in wordList:
            eachSentence.append(word)
        else:
            eachSentence=[]
            break
    #print(eachSentence)
    sentenceList.append(eachSentence)

sentenceList=[x for x in sentenceList if x]

print("word : ",len(wordList))
print("sentence : ",len(sentenceList)) 

output_filename1='talk_data/talk_sentence'+str(max_length)+'_with_border'+str(border)+'.json'
output_filename2='talk_data/talk_wordList'+str(max_length)+'_with_border'+str(border)+'.json'
with open(output_filename1,'w') as g:
    json.dump(sentenceList,g)

with open(output_filename2,'w') as h:
    json.dump(wordList,h)
