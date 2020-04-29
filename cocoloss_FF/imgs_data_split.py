#!/usr/bin/env python
import os
from os.path import join
import json
import itertools
import pickle

ori = '/home/aiiulab/Face_all/c40/original'
NT = '/home/aiiulab/Face_all/c40/NT'
F2F = '/home/aiiulab/Face_all/c40/F2F'
FS = '/home/aiiulab/Face_all/c40/FS'
DF = '/home/aiiulab/Face_all/c40/DF'
fakes = [NT, F2F, FS, DF]

files = ['test']

def main():
	for f in files:
		fr = json.load(open(f+'.json', 'r'))
		with open('/home/aiiulab/coco/jeromeTmp/FaceForensic/dataset/c40_all/'+f+'.csv', 'w') as fw:
			for pair in fr:
				# real
				for vid in pair:
					vid = join(ori, vid, '1')
					print(vid)
					for frm in os.listdir(vid):
						frm = join(vid, frm)
						fw.write(frm+',0\n')

				# fake
				for vid in list(itertools.permutations(pair, 2)):

					NTvid = join(NT, '_'.join(list(vid)))
					print(NTvid)
					with open('../../check_face_test/NT.pkl', 'rb') as f:
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
					with open('../../check_face_test/F2F.pkl', 'rb') as f:
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
					with open('../../check_face_test/FS.pkl', 'rb') as f:
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
					with open('../../check_face_test/DF.pkl', 'rb') as f:
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