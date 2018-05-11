from glob import glob
fonts=glob("/data/fonts_cn/*")

f=open("freqcn").read()
chars = list(set(list(f)))

letters=list(chr(i) for i in range(97,97+26))
letters2=list(chr(i).upper() for i in range(97,97+26))
numbers=list(str(i) for i in range(10))
chars = list(set(chars)-set(letters)-set(letters2)-set(numbers))
chars =numbers + letters + letters2 + chars


IDX2CHARS = dict((k,v) for k,v in enumerate(chars))
CHARS2IDX = dict((v,k) for k,v in enumerate(chars))
rg_n = (0,len(numbers))
rg_l = (rg_n[1],rg_n[1] + len(letters))
rg_u = (rg_l[1],rg_l[1] + len(letters2))
rg_c = (rg_u[1],rg_u[1] + len(chars)-len(numbers)-len(letters)-len(letters2))