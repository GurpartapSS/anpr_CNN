import keras
import cv2
import numpy as np
import time
import os
from keras.models import model_from_json
from os.path import splitext
from src.label import Label
from src.utils import getWH, nms, im2single
from src.projection_utils import getRectPts, find_T_matrix
from src.label import Shape, writeShapes


## Import pretrained WPO-Net model

json_file = open('source_model\wpod-net_update1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.save_weights('source_model\wpod-net_update1.h5')

## Read Input image ..currently only one image 

Im_path = r"images\test_C.jpg"
I = cv2.imread(Im_path)

lp_threshold = .5

ratio = float(max(I.shape[:2]))/min(I.shape[:2])
side  = int(ratio*288.)
bound_dim = min(side + (side%(2**4)),608)

def detect_lp(model,I,max_dim,net_step,out_size,threshold):

	min_dim_img = min(I.shape[:2])
	factor 		= float(max_dim)/min_dim_img

	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
	w += (w%net_step!=0)*(net_step - w%net_step)
	h += (h%net_step!=0)*(net_step - h%net_step)
	Iresized = cv2.resize(I,(w,h))

	T = Iresized.copy()
	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

	start 	= time.time()
	Yr 		= model.predict(T)
	Yr 		= np.squeeze(Yr)
	elapsed = time.time() - start

	L,TLps = reconstruct(I,Iresized,Yr,out_size,threshold)

	return L,TLps,elapsed


def reconstruct(Iorig,I,Y,out_size,threshold=.9):

	net_stride 	= 2**4
	side 		= ((208. + 40.)/2.)/net_stride # 7.75

	Probs = Y[...,0]
	Affines = Y[...,2:]
	rx,ry = Y.shape[:2]
	ywh = Y.shape[1::-1]
	iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))

	xx,yy = np.where(Probs>threshold)

	WH = getWH(I.shape)
	MN = WH/net_stride

	vxx = vyy = 0.5 #alpha

	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
	labels = []

	for i in range(len(xx)):
		y,x = xx[i],yy[i]
		affine = Affines[y,x]
		prob = Probs[y,x]

		mn = np.array([float(x) + .5,float(y) + .5])

		A = np.reshape(affine,(2,3))
		A[0,0] = max(A[0,0],0.)
		A[1,1] = max(A[1,1],0.)

		pts = np.array(A*base(vxx,vyy)) #*alpha
		pts_MN_center_mn = pts*side
		pts_MN = pts_MN_center_mn + mn.reshape((2,1))

		pts_prop = pts_MN/MN.reshape((2,1))

		labels.append(DLabel(0,pts_prop,prob))

	final_labels = nms(labels,.1)
	TLps = []

	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
		for i,label in enumerate(final_labels):

			t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
			ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
			H 		= find_T_matrix(ptsh,t_ptsh)
			Ilp 	= cv2.warpPerspective(Iorig,H,out_size,borderValue=.0)

			TLps.append(Ilp)

	return final_labels,TLps



class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,1)
		br = np.amax(pts,1)
		Label.__init__(self,cl,tl,br,prob)

Llp,LlpImgs,_ = detect_lp(loaded_model,im2single(I),bound_dim,2**4,(240,80),lp_threshold)

output_dir = r"images"
bname = splitext(os.path.basename(Im_path))[0]

if len(LlpImgs):
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

				s = Shape(Llp[0].pts)

				cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])