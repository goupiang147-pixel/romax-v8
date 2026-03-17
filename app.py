#!/usr/bin/env python3

from flask import Flask, jsonify
import requests
import time
import random
import math
import os
import numpy as np
from collections import Counter, deque
from datetime import datetime
import threading

app = Flask(__name__)

API="https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

# =========================
# UTIL
# =========================
def get_size(n):
    return "BIG" if n>=5 else "SMALL"

# =========================
# TERMINAL LOG
# =========================
class Terminal:
    def __init__(self):
        self.logs=deque(maxlen=40)

    def log(self,msg):
        t=datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{t}] {msg}")

# =========================
# KALMAN
# =========================
def kalman_filter(seq):
    q=0.05
    r=0.4
    x=seq[0]
    p=1
    out=[]
    for z in seq:
        p=p+q
        k=p/(p+r)
        x=x+k*(z-x)
        p=(1-k)*p
        out.append(x)
    return out

# =========================
# ENGINES
# =========================
class DragonPPM:
    def predict(self,h):
        if len(h)<20: return None
        s=[get_size(x) for x in h]
        for depth in range(7,2,-1):
            p=s[-depth:]
            b=0; sm=0
            for i in range(len(s)-depth-1):
                if s[i:i+depth]==p:
                    if s[i+depth]=="BIG": b+=1
                    else: sm+=1
            if b>sm: return "BIG"
            if sm>b: return "SMALL"
        return None

class Markov:
    def predict(self,h):
        if len(h)<25: return None
        s=[get_size(x) for x in h]
        trans={"BIG":{"BIG":0,"SMALL":0},"SMALL":{"BIG":0,"SMALL":0}}
        for i in range(len(s)-1):
            trans[s[i]][s[i+1]]+=1
        last=s[-1]
        if trans[last]["BIG"]>trans[last]["SMALL"]: return "BIG"
        if trans[last]["SMALL"]>trans[last]["BIG"]: return "SMALL"
        return None

class Bayesian:
    def predict(self,h):
        if len(h)<50: return None
        last=h[-50:]
        big=sum(1 for x in last if x>=5)
        p=big/50
        if p>0.56: return "BIG"
        if p<0.44: return "SMALL"
        return None

class Fractal:
    def predict(self,h):
        if len(h)<30: return None
        s=[get_size(x) for x in h]
        for k in range(2,6):
            p=s[-k:]
            m=0
            for i in range(len(s)-k*2):
                if s[i:i+k]==p: m+=1
            if m>=2: return p[0]
        return None

class Chaos:
    def predict(self,h):
        if len(h)<40: return None
        seq=[1 if x>=5 else 0 for x in h[-40:]]
        p=sum(seq)/len(seq)
        ent=-(p*math.log2(p+1e-6)+(1-p)*math.log2(1-p+1e-6))
        if ent<0.9:
            return "BIG" if p>0.5 else "SMALL"
        return None

class NeuralMatrix:
    def __init__(self):
        self.w=np.random.randn(20)
    def predict(self,h):
        if len(h)<20: return None
        seq=[1 if x>=5 else 0 for x in h[-20:]]
        seq=kalman_filter(seq)
        s=np.dot(seq,self.w)/len(seq)
        return "BIG" if s>0 else "SMALL"

class Attention:
    def predict(self,h):
        if len(h)<25: return None
        seq=[1 if x>=5 else 0 for x in h[-25:]]
        att=np.exp(seq)/np.sum(np.exp(seq))
        score=sum(a*s for a,s in zip(att,seq))
        return "BIG" if score>0.5 else "SMALL"

class Sequence:
    def predict(self,h):
        if len(h)<20: return None
        seq=[1 if x>=5 else 0 for x in h[-20:]]
        w=np.linspace(0.1,1,len(seq))
        score=np.dot(seq,w)/w.sum()
        return "BIG" if score>0.5 else "SMALL"

class Momentum:
    def predict(self,h):
        if len(h)<15: return None
        seq=[1 if x>=5 else 0 for x in h[-15:]]
        diff=sum(seq[i]-seq[i-1] for i in range(1,len(seq)))
        return "BIG" if diff>0 else "SMALL"

class Trend:
    def predict(self,h):
        if len(h)<20: return None
        seq=[1 if x>=5 else 0 for x in h[-20:]]
        x=np.arange(len(seq))
        slope=np.polyfit(x,seq,1)[0]
        return "BIG" if slope>0 else "SMALL"

class Frequency:
    def predict(self,h):
        if len(h)<30: return None
        s=[get_size(x) for x in h[-30:]]
        c=Counter(s)
        return "BIG" if c["BIG"]>c["SMALL"] else "SMALL"

class Alternation:
    def predict(self,h):
        if len(h)<10: return None
        s=[get_size(x) for x in h[-10:]]
        alt=sum(1 for i in range(1,len(s)) if s[i]!=s[i-1])
        return "BIG" if alt<5 else "SMALL"

class MeanReversion:
    def predict(self,h):
        if len(h)<30: return None
        avg=sum(h[-30:])/30
        return "BIG" if avg<4.5 else "SMALL"

class Cluster:
    def predict(self,h):
        if len(h)<20: return None
        seq=[1 if x>=5 else 0 for x in h[-20:]]
        cluster=sum(seq[-5:])
        return "BIG" if cluster>=3 else "SMALL"

class RandomForestLite:
    def __init__(self):
        self.weights=np.random.rand(10)
    def predict(self,h):
        if len(h)<10: return None
        seq=[1 if x>=5 else 0 for x in h[-10:]]
        score=np.dot(seq,self.weights)/len(seq)
        return "BIG" if score>0.5 else "SMALL"

# =========================
# AI CORE
# =========================
class UltraAI:

    def __init__(self,term):
        self.history=deque(maxlen=200000)
        self.term=term

        self.engines={
            "dragon":DragonPPM(),
            "markov":Markov(),
            "bayes":Bayesian(),
            "fractal":Fractal(),
            "chaos":Chaos(),
            "neural":NeuralMatrix(),
            "attention":Attention(),
            "sequence":Sequence(),
            "momentum":Momentum(),
            "trend":Trend(),
            "frequency":Frequency(),
            "alternation":Alternation(),
            "meanrev":MeanReversion(),
            "cluster":Cluster(),
            "forest":RandomForestLite()
        }

    def add(self,n):
        self.history.append(n)

    def predict(self):

        h=list(self.history)
        votes=[]

        for name,eng in self.engines.items():
            p=eng.predict(h)
            if p:
                for _ in range(3):
                    votes.append(p)

        if not votes:
            return random.choice(["BIG","SMALL"]),50

        c=Counter(votes)
        pred=c.most_common(1)[0][0]
        conf=int((c[pred]/len(votes))*100)

        return pred,conf

# =========================
# FETCH
# =========================
def fetch():
    try:
        r=requests.get(API,timeout=10)
        d=r.json()
        latest=d["data"]["list"][0]
        return latest["issueNumber"],int(latest["number"])
    except:
        return None,None

# =========================
# GLOBAL DATA
# =========================
ai = UltraAI(Terminal())

data={
    "period":"-",
    "number":0,
    "prediction":"-",
    "confidence":0,
    "wins":0,
    "losses":0
}

# =========================
# WORKER
# =========================
def worker():

    last=None
    last_pred=None
    last_backup=[]

    wins=0
    losses=0

    while True:
        try:
            period,num=fetch()

            if period and period!=last:

                ai.add(num)

                if last_pred:
                    if get_size(num)==last_pred:
                        wins+=1
                    elif num in last_backup:
                        wins+=1
                    else:
                        losses+=1

                pred,conf=ai.predict()

                if pred=="BIG":
                    backup=random.sample([0,1,2,3,4],2)
                else:
                    backup=random.sample([5,6,7,8,9],2)

                data.update({
                    "period":period,
                    "number":num,
                    "prediction":pred,
                    "confidence":conf,
                    "wins":wins,
                    "losses":losses
                })

                last_pred=pred
                last_backup=backup
                last=period

        except:
            pass

        time.sleep(2)

threading.Thread(target=worker,daemon=True).start()

# =========================
# API
# =========================
@app.route("/")
def home():
    return jsonify(data)

# =========================
# RUN
# =========================
if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",3000)))