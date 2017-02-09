import json
import MeCab 
import re
import mojimoji
#Q,Aのディクショナリが入ったリストを前処理
#不要な単語を捨て、原型だけでできたリストに変換
with open('2010_2016_navi_qa.json') as f:
    data=json.load(f)


tagger=MeCab.Tagger("-d /usr/lib/mecab/dic/mecab-ipadic-neologd/")

QAList=[]

i=0
for parts in data:
    print(i)
    q_=''
    a_=''
    Q=[]
    A=[]
    i=i+1
    q=tagger.parse(re.sub('\?','？',mojimoji.zen_to_han(re.sub(r'https?://([a-zA-Z0-9!-~]*)+?','',parts['q']),kana=False)))
    q=q.split('\n')
    for part in q:
        if part=='EOS':
            break
        else:
            surface,features=part.split('\t')
            features=features.split(',')
            if re.match(r"[a-zA-Z]",surface[0]):
                q_+=features[-1]
            else:
                q_+=surface
    q_=re.sub("\[.*?\]|\{.*?\}|\[\[.*?\]\]|\(.*?\)|（.*?）|｛.*?｝|==.*==|[a-zA-Z]|この質問は.*?リクエストしました。","",q_)
    q=tagger.parse(q_)
    q=q.split('\n')
    for part in q:
        if part=='EOS':
            break
        else:
            surface,features=part.split('\t')
            features=features.split(',')
            if features[1]=='数':
                Q.append(surface)
            elif features[0]=='名詞' and features[-1]!='カオモジ':
                Q.append(surface if features[-3]=='*' else features[-3])
            elif (features[0]!='記号' or surface=='。' or surface=='？' or surface=='、') and features[-1]!='カオモジ':
                #Q.append(surface if re.match(r'\*|[a-z0-9A-Z]|[ａ-ｚ０-９Ａ-Ｚ]',features[-3]) else features[-3])#features[-3]は原型
                
                Q.append(surface)

    q=tagger.parse(re.sub('\?','？',mojimoji.zen_to_han(re.sub(r'https?://([a-zA-Z0-9!-~]*)+?','',parts['a']),kana=False)))
    q=q.split('\n')
    for part in q:
        if part=='EOS':
            break
        else:
            surface,features=part.split('\t')
            features=features.split(',')
            if re.match(r"[a-zA-Z]",surface[0]):
                a_+=features[-1]
            else:
                a_+=surface
    a_=re.sub("\[.*?\]|\{.*?\}|\[\[.*?\]\]|\(.*?\)|（.*?）|｛.*?｝|==.*==|[a-zA-Z]","",a_)
    q=tagger.parse(a_)
    q=q.split('\n')
    for part in q:
        if part=='EOS':
            break
        else:
            surface,features=part.split('\t')
            features=features.split(',')
            if features[1]=='数':
                A.append(surface)
            elif features[0]=='名詞' and features[-1]!='カオモジ':
                A.append(surface if features[-3]=='*' else features[-3])
            elif (features[0]!='記号' or surface=='。' or surface=='？' or surface=='、' )and features[-1]!='カオモジ':
                #A.append(surface if re.match(r'\*|[a-z0-9A-Z]|[ａ-ｚ０-９Ａ-Ｚ]',features[-3]) else features[-3])
                A.append(surface)
    if i%200==0:
        print('before:\n',parts['q'])
        print('after:\n',Q)
        print('before:\n',parts['a'])
        print('after:\n',A)
        
    #QAList.append(Q)
    #QAList.append(A)
    QAList.append({'q':Q,'a':A})
with open('genkei_dic_navi2010_chie.json','w') as g:
    json.dump(QAList,g)
