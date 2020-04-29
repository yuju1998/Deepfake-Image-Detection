#!/usr/bin/env python
import os
from os.path import join
import json
import itertools
import pickle

ori = '/home/frank/center/c40/real'
NT = '/home/frank/center/c40/NT'
F2F = '/home/frank/center/c40/F2F'
FS = '/home/frank/center/c40/FS'
DF = '/home/frank/center/c40/DF'
fakes = [NT, F2F, FS, DF]
files = ['train', 'val']

def main():
	for f in files:
		fr = json.load(open(f+'.json', 'r'))
		with open(f+'.csv', 'w') as fw:
			for pair in fr:
				# real
				for vid in pair:
					vid = join(ori, vid)
					print(vid)
					for frm in os.listdir(vid):
						frm = join(vid, frm)
						fw.write(frm+',0\n')

				# fake
				for vid in list(itertools.permutations(pair, 2)):

					NTvid = join(NT, '_'.join(list(vid)))
					print(NTvid)
					with open('../../check/NT.pkl', 'rb') as f:
						NTlm = pickle.load(f)

					for frm in os.listdir(NTvid):
						try:
							if NTlm[NTvid.split('/')[-1]][frm.split('.')[0]]:
								frm = join(NTvid, frm)
								fw.write(frm+',1\n')
						except:
							print("ignore")


					F2Fvid = join(F2F, '_'.join(list(vid)))
					print(F2Fvid)
					with open('../../check/F2F.pkl', 'rb') as f:
						F2Flm = pickle.load(f)
					
					for frm in os.listdir(F2Fvid):
						try:
							if F2Flm[F2Fvid.split('/')[-1]][frm.split('.')[0]]:
								frm = join(F2Fvid, frm)
								fw.write(frm+',2\n')
						except:
							print("ignore")


					FSvid = join(FS, '_'.join(list(vid)))
					print(FSvid)
					with open('../../check/FS.pkl', 'rb') as f:
						FSlm = pickle.load(f)

					for frm in os.listdir(FSvid):
						try:
							if FSlm[FSvid.split('/')[-1]][frm.split('.')[0]]:
								frm = join(FSvid, frm)
								fw.write(frm+',3\n')
						except:
							print("ignore")


                                  
					DFvid = join(DF, '_'.join(list(vid)))
					print(DFvid)
					with open('../../check/DF.pkl', 'rb') as f:
						DFlm = pickle.load(f)
						
					for frm in os.listdir(DFvid):
						try:
							if DFlm[DFvid.split('/')[-1]][frm.split('.')[0]]:
								frm = join(DFvid, frm)
								fw.write(frm+',4\n')
						except:
							print("ignore")


if __name__=='__main__':
	main()