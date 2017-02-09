import json 
import re
import urllib.request
import time
from bs4 import BeautifulSoup



def scraping(Qid,QAList):
    url='http://detail.chiebukuro.yahoo.co.jp/qa/question_detail/'+Qid
    html=urllib.request.urlopen(url).read()
    html=html.strip().decode("utf-8").replace('\n','')
    soup=BeautifulSoup(html,'lxml')

    #soup_usr=soup.find('div',class_='usrQstn')
    #usr_txt=re.sub(r'\t','',soup_usr.find('p',class_='queTxt').getText())
    #usr_txt_list=soup.find('div',class_='usrQstn').getText().split('共感した')[0:-1]
    #usr_txt=re.search(r'\t.*?共感した',usr_txt).group()
    #usr_txt=''
    #for part in usr_txt_list:
    #usr_txt=usr_txt+part
    #usr_txt=re.sub(r'\t','',re.search(r'\t.*',usr_txt).group())
    usr_txt=soup.find('div',class_='ptsQes').getText()
    usr_sub=soup.find('div',class_='attInf').getText()
    usr_txt=re.sub(usr_sub,'',usr_txt)
    usr_txt=re.sub(r'\t|\n|\r','',usr_txt)
    soup_BA1=soup.find('div',class_='mdPstd mdPstdBA othrAns clrfx')
    soup_BA2=soup.find('div',class_='mdPstd mdPstdBA othrAns lstLast clrfx')
    if soup_BA1==None:
        BA_txt=soup_BA2.find('div',class_='ptsQes').getText()
        BA_sub=soup_BA2.find('div',class_='attInf').getText()
        BA_txt=re.sub(BA_sub,'',BA_txt)
        BA_txt=re.sub('\t|\n|\r','',BA_txt)
    else:
        BA_txt=soup_BA1.find('div',class_='ptsQes').getText()
        BA_sub=soup_BA1.find('div',class_='attInf').getText()
        BA_txt=re.sub(BA_sub,'',BA_txt)
        BA_txt=re.sub('\t|\n|\r','',BA_txt)
    QAList.append({'q':usr_txt,'a':BA_txt,'q_id':Qid})
    time.sleep(1.0)
    #print(QAList)

for year in range(2004,2017):
    print('year : ',year)
    for month in range(1,13):
        if year==2004 and month<4:
            continue
        print('month : ',month)
        QAList=[]
        t=""
        i=1
        filename='chiebukuro'+str(month)+'_'+str(year)+'.json'
        with open(filename) as f:
            data=json.load(f)
        for part in data:
            t=part['url']
            try:
                scraping(t,QAList)
                print(i,'   finished')
                i=i+1
                
            except:
                print(t)
                for j in range(10):
                    try:
                        time.sleep(5.0)
                        scraping(t,QAList)
                        print(i,'   finished')
                        break
                    except:
                        print('error')
                

        output_filename='all_qadata/chie_data'+str(month)+'_'+str(year)+'.json'
        with open(output_filename,'w') as g:
            if len(QAList)>0:
                json.dump(QAList,g)
                print('finish')
