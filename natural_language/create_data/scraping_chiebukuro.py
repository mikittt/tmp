# -*- coding:utf-8 -*-

import urllib.request
from bs4 import BeautifulSoup
import re
import json
import time


for year in range(2012,2017):
    print('year : ',year)
    for i in range(1,13):
        obj=[]
        if year==2012 and i<9:
            continue
        filename='chiebukuro'+str(i)+'_'+str(year)+'.json'
        print('month : ',i)
        if i==4 or i==6 or i==9 or i==11:
            day_max=30
        elif i==2:
            if year % 4 ==0:
                day_max=29
            else:
                day_max=28
        else:
            day_max=31
        day_max=day_max+1
        for j in range(1,day_max):
            print('day : ',j)
            k=1
            while(1):
                try:
                    html=urllib.request.urlopen("http://chiebukuro.yahoo.co.jp/dir/list.php?did=2078297908&flg=1&sort=3&type=list&year=%d&month=%d&day=%d&page=%d"%(year,i,j,k)).read()
                    html=html.strip().decode("utf-8").replace('\n','')
                    matchOB=re.search(r'<ul id="qalst" data-ult_sec="1">.*</ul>',html)
                    soup=BeautifulSoup(matchOB.group(),'lxml')
                    qList=soup.findAll("dt")
                    for part in qList:
                        obj.append({"url":part.find("a")['href'].split('/')[-1],"title":part.getText()})
                    time.sleep(1.0)
                    print('finished page : ',k)
                    k=k+1
                
                except:
                    break
        with open(filename,'w') as f:
            json.dump(obj,f)

#with open("chiebukuro_syusyoku2004.json","w") as f:
    #json.dump(obj,f)

