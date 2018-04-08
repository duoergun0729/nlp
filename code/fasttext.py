# -*- coding: UTF-8 -*-
import re

def load_SogouTCE():
    SogouTCE=[]
    SogouTCE_kv = {}
    with open("../data/SogouTCE.txt") as F:
        for line in F:
            (url,channel)=line.split()
            SogouTCE.append(url)
        F.close()

    for index,url in enumerate(SogouTCE):
        print "k:%s v:%d" % (url,index)
        SogouTCE_kv[url]=index

    return  SogouTCE_kv

def load_url(SogouTCE_kv):
    with open("../data/news_sohusite_url.txt") as F:
        for line in F:
            for k,v in SogouTCE_kv.items():
                if re.match(k,line,re.IGNORECASE):
                    print "x:%s y:%d" % (line,v)

        F.close()

if __name__ == '__main__':
    SogouTCE_kv=load_SogouTCE()


    load_url(SogouTCE_kv)