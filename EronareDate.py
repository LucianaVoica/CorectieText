# Construirea sursei - Contine date eronate in mod cotrolat (FINAL)
import torch
import spacy
from PrSpacy import Romana
import  re
import numpy as np
import random
import jiwer

nlp = Romana()
g=open("file2_eronate.txt", mode = 'w',encoding = 'utf-8')

# Construiesc setul de date in care introduc in mod controlat erori
medie = 0.4 
secventa = np.linspace(0.01, 0.015)
abatere = random.choice(secventa)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Functii ce introduc erorile
# inlocuire de diacritica ( sau cu diacritica)
def inlocuire_diacritica(word): #Verificata
    for i in range(len(word)):
        if word[i] == 'ă':
            word=re.sub(str(word[i]),'a',str(word))
        if word[i] == 'â':
            word=re.sub(str(word[i]),'a',str(word))
        if word[i] == 'ș':
            word=re.sub(str(word[i]),'s',str(word))
        if word[i]== 'ț':
            word=re.sub(str(word[i]),'t',str(word))
        if word[i] == 'î':
            word=re.sub(str(word[i]),'i',str(word))
    return word

def inlocuire_cuDiacritica(word): # Verificata
    for i in range(len(word)):
        if word[i] == 'a': 
            r=random.randint(1,2) #se alege random daca a va f inlocuit cu  ă sau â
            if r == 1:
                word=re.sub(str(word[i]),'ă',str(word))
            else:
                word=re.sub(str(word[i]),'â',str(word))
        if word[i] =='s':
            word=re.sub(str(word[i]),'ș',str(word))
        if word[i] =='t':
            word=re.sub(str(word[i]),'ț',str(word))
        if word[i] == 'i':
            word=re.sub(str(word[i]),'î',str(word))
    return word

# inlocuire semn de punctuatie
def inlocuire_semn(word):
    semne = [';',':','!']
    i=random.randint(0,len(semne)-1)
    return word.replace(word,semne[i],1)

def eronare(propozitii):
    nr=40  # nr de propozitii din lista data ca parametru
    valori = np.linspace(medie-abatere, medie+abatere)
    p_eronate = []
    for i in range(nr):
        rata_impusa = round(random.choice(valori),2) # se alege random rata de eronare a unei propozitii
        prop1=nlp(propozitii[i])    # rămâne nemodificat, cu el se face comparația
        prop2=nlp(propozitii[i])
        while True:
            rata_eroare=round(jiwer.wer(prop1,prop2),2) #verificam cat de eronta este propozitia modificata
            if rata_eroare>rata_impusa:
                break # Nu vrem sa avem o eroare prea mare la niv unei prop, pt ca reteaua nu va mai face fata
            else:
                index=random.randint(0,len(prop2)-1) # se alege indexul unui cuvant/semn random pt a fi eronat
                word=prop2[index]
                semn=[',','.','?'] # doar aceste semne se gasesc in baza mea de date
                ok=1
                for j in range(3):  # Vedem daca e cuvant sau semn de punctuatie
                    if str(word)==semn[j]:
                        ok=0
                if ok:
                    p=random.randint(0,1) # alegem in ce fel eronam cuvantul
                    if p==0:
                        word_eronat=inlocuire_diacritica(str(word))
                        prop2=re.sub(str(prop2[index]),word_eronat, str(prop2))
                    if p==1:
                        word_eronat=inlocuire_cuDiacritica(str(word))
                        prop2=re.sub(str(prop2[index]),word_eronat, str(prop2))
                else:
                    p=random.randint(0,1) # alegem in ce fel eronam semnul
                    if p==0: 
                        semn_eronat=inlocuire_semn(str(word))
                        prop2=re.sub(str(prop2[index]),semn_eronat, str(prop2))
                        
                    if p==1: 
                        prop2=re.sub(str(word),"",1)
                
        g.write(prop2)
   
    

                    


# Constructia fisier sursa cu propozitii gresite ( in mod controlat)
k=0
t=0
s=0
propozitii = []
with open("file2.txt",'r', encoding = 'utf-8') as f:
    Lines = f.readlines()
for line in Lines:
    # print(line)
    k += 1     
    t += 1
    propozitii.append(line.lower())
    if k==40:   # aplic functia de eronare pe seturi de cate 50 propozitii 
                # nlp ( spacy) nu lucreaza cu f multe caractere 
                # si de aceea nu am aplicat functia pe tot setul de date
        eronare(propozitii)
        k=0
        propozitii = []
       