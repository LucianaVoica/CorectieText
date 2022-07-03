# Curatarea textul extras din doc web. 
from PrSpacy import Romana
import  re
nlp = Romana()
f=open("date.txt", mode = 'r',encoding = 'utf-8')
g=open("date_C.txt", mode = 'w', encoding = 'utf-8')
s=f.read()
elimin = [' alin.',' nr.', ' art.',' Art.','Notă ',' pct.','Alin.',' lit.',' Lit.', ' HP','volume_upmore_vertRomanian','volume_up',' dvs.', ' int ']
s=s.translate({ord(elimin[i]):None for i in range(len(elimin))})
s=s.translate({ord(i):None for i in '"←0123456789=–«»">>&^”‘’:;_()[\n\t]+%*'})
s = re.sub('\t','. ',s)
s = re.sub('!','.',s)
s = re.sub(' .','.',s)
s = re.sub('$NE$','',s)
g.write(s)