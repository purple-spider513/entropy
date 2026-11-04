#!/usr/bin/env python3
"""
This script is used to get element composition from all_atom_type.csv
Input and out files need to be defined.
"""

import os, re
fo=open("train_atom_mw300.csv","r")     ####atom-types
lig_list=fo.readlines()
out=open("train_element_mw300.csv","w") ####element-types
out.write("ID,ring,heavy,s_heavy_bond,MW,B,Br,C,Cl,F,Fe,H,I,N,O,P,S,Si,symmetry,entropy\n")

n=0
for line in lig_list:
    if n>=1:
        ln=line.split(",")
        C=int(ln[7])+int(ln[8])+int(ln[9])+int(ln[10])+int(ln[11])
        N=int(ln[17])+int(ln[18])+int(ln[19])+int(ln[20])+int(ln[21])+int(ln[22])+int(ln[23])
        O=int(ln[24])+int(ln[25])+int(ln[26])
        S=int(ln[28])+int(ln[29])+int(ln[30])+int(ln[31])+int(ln[32])+int(ln[33])
        ent=float(ln[36])
        out.write(f"{ln[0]},{ln[1]},{ln[2]},{ln[3]},{ln[4]},{ln[5]},{ln[6]},{C},{ln[12]},{ln[13]},{ln[14]},{ln[15]},{ln[16]},{N},{O},{ln[27]},{S},{ln[34]},{ln[35]},{ent}\n")
    n+=1
    if n%100==0:
        print(f"{n}/{len(lig_list)}")
print(f"Total {n} ligands are processed!")
fo.close()
out.close()
exit
