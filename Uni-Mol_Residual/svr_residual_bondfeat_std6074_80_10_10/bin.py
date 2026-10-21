#!/usr/bin/python3
#This script is used to generate bins for relative entropy deviation

maxBin=7  #8  #14, 11
minBin=-8  #-8 #-19, -8

fo=open("test.dat","r")
#fo=open("std_corr.dat","r")
data=fo.readlines()

out=open("test_bins.dat","w")
#out=open("test_bins","w")

n=0
for i in range(minBin,maxBin):
    bin_low=i
    bin_high=i+1
    v=(bin_low+bin_high)/2
    temp=0
    for line in data:
        line=line.strip()
        ln=line.split()
        if float(ln[3])>=bin_low and float(ln[3])<bin_high:
            temp +=1
    out.write(f"{round(v,1)}    {temp}\n")
fo.close()
out.close()
