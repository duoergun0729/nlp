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
        #删除http前缀
        url=re.sub('http://','',url)
        print "k:%s v:%d" % (url,index)
        SogouTCE_kv[url]=index

    return  SogouTCE_kv

def load_url(SogouTCE_kv):
    labels=[]
    with open("../data/news_sohusite_url.txt") as F:
        for line in F:
            for k,v in SogouTCE_kv.items():
                if re.search(k,line,re.IGNORECASE):
                    #print "x:%s y:%d" % (line,v)
                    print v
                    labels.append(v)
                #else:
                #    print "not found %s" %(line)

        F.close()
    return  labels

if __name__ == '__main__':
    SogouTCE_kv=load_SogouTCE()

    labels=load_url(SogouTCE_kv)