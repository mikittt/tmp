import json
#単語リストを保存
filename='raw_mynavi_chie_withBorder_100.json'

with open(filename) as f:
    data=json.load(f)


words=[]
for part in data:
    words.extend(part)
words=list(set(words))


output_filename='raw_mynavi_chie_100_words.json'

with open(output_filename,'w') as g:
    json.dump(words,g)
