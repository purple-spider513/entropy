#!/usr/bin/python3
"""
This script is used to generate atom feature plus general descriptors
for calssical machine learning training and prediction.
It needs four input data: predifined bond features, DFT_entropy,
Conformer-symmetry_correction, mol2_file, and one output file. They 
are defined in this script.
"""

import os, re, math

fo=open("unique_atom_type","r")         ####predifined atom type library
atom_list=fo.readlines()
atom_list.sort()

fo_dft=open("mw300_DFT","r")            #### DFT calculation result
dft_list=fo_dft.readlines()
dict_dft={}  #DFT entropy
for i in dft_list:
    i=i.strip()
    dict_dft[i.split()[0]]=i.split()[1]

out=open("train_atom_mw300.csv","w")    #### Output file name
out.write("ID,ring,heavy,s_heavy_bond,MW,")

for lig in atom_list:
    lig=lig.strip()
    out.write(f"{lig},")
out.write("symmetry,entropy\n")
fo.close()

dict_csv={}  #for all atom types

#Read entropy values and mol2 files from gauss calculation
fo1=open("mw300_conformers-symmetry.csv")####conformer-symmetry correction data
conf_list=fo1.readlines()

n=0
for line in conf_list: 
    #initialize atom type count
    for atom in atom_list:
        atom=atom.strip()
        dict_csv[atom]=0

    line=line.strip()
    ln=line.split(",")
    if re.search("molecule_id",line) or re.search("error",line):
        continue
	
    dft=float(dict_dft[ln[0]])
    symmetry=float(ln[8])
    val=symmetry+dft
    lig=ln[0]+".mol2"
    try:
        fo2=open(f"mw300_mol2/{lig}","r")  #### Mol2 files
        foo2=fo2.readlines()
    except FileNotFoundError:
        continue
    flag=0
    n+=1
    lst=foo2[2].split()  ####pdbbind is foo[3]
    ring=int(lst[1])-int(lst[0])+1
    heavy=0 #heavy atom number
    s_heavy_bond=0 #single heavy bond
    MW=0    #molecular weight
    H=0     #proton number
    dict_atom={}  #save the current lig atom ID and atom type pair
    for line2 in foo2:
        if re.match("@<TRIPOS>MOLECULE",line2):
            flag=1
        if re.match("@<TRIPOS>ATOM",line2):
            flag=2
        if re.match("@<TRIPOS>BOND",line2):
            flag=3
        if re.match("@<TRIPOS>SUBSTRUCTURE",line2):
            break
        if flag==2 and line2[72:78]:
            ln2=line2.split()
            M=ln2[5].split(".")[0]
            dict_atom[ln2[0]]=ln2[5]
            if re.match("H",ln2[1]):
#                H+=1
                MW+=1.008  # MW for H
            else:
                heavy+=1
                if M=="B":
                    MW+=10.81
                elif M=="Cl" or ln2[1]=="CL":
                    MW+=35.453
                elif M=="Br" or ln2[1]=="BR":
                    MW+=79.904
                elif M=="Si" or ln2[1]=="SI":
                    MW+=28.085
                elif M=="C":
                    MW+=12.011
                elif M=="O":
                    MW+=15.999
                elif M=="S":
                    MW+=32.06
                elif M=="N":
                    MW+=14.007
                elif M=="P":
                    MW+=30.974
                elif M=="F":
                    MW+=18.998
                elif M=="I":
                    MW+=126.90
                elif M=="Fe" or ln2[1]=="FE":
                    MW+=55.845
                else:
                    pass
            try:
                dict_csv[ln2[5]]+=1
            except KeyError:
                print(line)
        if flag==3 and line2[16:18]:  #for bond type check
            ln3=line2.split()
#            print(ln3,lig)
            if ln3[3]=='1':
                if dict_atom[ln3[1]]!='H' and dict_atom[ln3[2]]!='H':
                    s_heavy_bond+=1
    MW=round(MW,3)
    out.write(f"{lig},")
#    print("yes")
    out.write(f"{ring},{heavy},{s_heavy_bond},{MW},")
    for keys in dict_csv:
        keys=keys.strip("\n")
        out.write(f"{dict_csv[keys]},")
    out.write(f"{symmetry},{val}\n")  #symmetry, and corrected entropy
    fo2.close()
    if n%100==0:
        print(f"{n}/{len(conf_list)}")
print(f"total successiful {n} mols")
out.close()
exit
