import numpy as np 
import cv2

import argparse
import sys
from pdb import set_trace

class BGModel:
	'''
	Method 1: Running Weighted Mean to compute background
	Method 2: TBD - Gaussian Mixture Models per pixel (Takes more time to write .. :/)
	'''
	model = []
	var  = []

	def __init__(self, frame_0, rho, threshold):
		(h,w,c) = frame_0.shape
		self.th  = threshold 
		self.rho = rho
		self.model = frame_0

		# initlaise better (based on neighbouring window)
		self.var = np.ones((h,w,c)).astype(np.float64) * 1e-1 
	
	def bgTrain(self, frame):

		diff = cv2.absdiff(self.model.astype(np.uint8), frame)
		self.model = self.rho*(frame) + (1-self.rho)*(self.model.astype(np.uint8))
		
		## ISSUE -> Varience explodes!
		# self.var  = self.rho*(diff**2) + (1-self.rho)*(self.var**2)
		# # Control varience eh?
		# self.var[self.var < 1e-10] = 1e-10
		# self.var[self.var > 1e+10] = 1e+10

	def getForeground(self, frame):
		
		## ISSUE -> Varience explodes!
		# diffs = cv2.absdiff(self.model.astype(np.uint8),frame) / self.var
		# mask = diffs > (self.th * self.var)

		# Simpler 
		diffs = cv2.absdiff(self.model.astype(np.uint8),frame)
		self.model = self.rho*(frame) + (1-self.rho)*(self.model.astype(np.uint8))
		
		ret, mask = cv2.threshold(diffs, self.th, 255, cv2.THRESH_BINARY)

		return mask

	def showModel(self):
		cv2.imshow('bgModel',self.model.astype(np.uint8))
		cv2.waitKey(0)


def videoWrite(video_path, images, width, height):
	video = cv2.VideoWriter(video_path,
		# cv2.VideoWriter_fourcc(*'MJPG'),
		cv2.VideoWriter_fourcc(*'MP4V'),
		20.0,
		(width,height))
	for img in images:
		video.write(np.array(img, dtype=np.uint8))
	
	video.release()

def plotBbox(img, msk, bbox, bigBlobIdx, scaleFactor):
	for idx in bigBlobIdx:
		# set_trace()
		this_object =  (msk==int(idx))
		x,y,w,h = cv2.boundingRect(this_object.astype(np.uint8))
		x,y,w,h = (np.asarray([x,y,w,h])/scaleFactor).astype(np.int)
		cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,255),2)
	return img

def preProcess(img, width, scaleFactor):

	# Covert to GrayScale
	# img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Rscale to width	
	img = cv2.resize(img, None, fx = scaleFactor, 
						fy = scaleFactor, 
						interpolation = cv2.INTER_AREA)
	
	# Gaussian Blurr
	img = cv2.medianBlur(img , 3)
	img = cv2.GaussianBlur(img, (3, 3), 0)

	return img.astype(np.uint8)

def main(vid, out, frameLimit, width, minBlobSize, rho, threshold):
	cap = cv2.VideoCapture(vid)
	results = []
	resDes = []	
	connectivity = 8

	ret, frame_0 = cap.read()
	if not ret:
		print("[ERROR] Video has no frame")
		sys.exit(1)	

	''' First Frame '''
	scaleFactor = width / float(frame_0.shape[1]) 
	frame_0 = preProcess(frame_0, width, scaleFactor)
	BGM = BGModel(frame_0, rho, threshold)
	frame_idx = 0

	while True:
		ret, frame = cap.read()
		frame_idx += 1
		
		if frame_idx < frameLimit:
			if not ret:
				print("Error! Video does not have enough frames to build BG Model")
				sys.exit(1)
			
			'''Compute weighted background mean over inital N frames'''
			res = frame.copy()
			frame = preProcess(frame, width, scaleFactor)
			BGM.bgTrain(frame)	
			cv2.imshow('res',res.astype(np.uint8))

		else:
			if not ret:
				break

			res = frame.copy()

			'''Background Subtraction'''
			frame = preProcess(frame, width, scaleFactor)	
			mask = BGM.getForeground(frame)
			
			'''Some Opening-Closing noise reduciton'''
			''' Doesn't help much'''
			kernel = np.ones((5,5),np.uint8)
			for i in range(4):
				mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
			for i in range(4):		
				mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

			'''Thresholding'''
			graymask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
			# ret,thresh1 = cv2.threshold(graymask,100,255,cv2.THRESH_BINARY)
			# th3 = cv2.adaptiveThreshold(graymask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
			
			'''Bolob Detection based on connected component mask'''
			stats = cv2.connectedComponentsWithStats(graymask, connectivity, cv2.CV_32S)
			_ , components_msk, bboxes, _ = stats
			(w,h) = components_msk.shape
			'''select blobs greater than min blob size and less than 2/3 of image size'''
			bigBlobs = np.logical_and( bboxes[:,4]>minBlobSize, bboxes[:,4]< (w*h*2)/3)
			bigBlobIdx = bigBlobs.nonzero()[0]
			plotBbox(res, components_msk, bboxes, bigBlobIdx, scaleFactor)

			plotBbox(mask, components_msk, bboxes, bigBlobIdx, 1.0)
			cv2.imshow('Foreground',mask.astype(np.uint8))
			resDes.append(mask)

			cv2.imshow('res',res.astype(np.uint8))
			# set_trace()

		'''Display Results'''
		key = cv2.waitKey(20) & 0xFF
		# key = cv2.waitKey(0)
		results.append(res)
		
	# print("Saving video as", out)
	# videoWrite(out, results, results[0].shape[1], results[0].shape[0])

def testVideoLoad(path):
	cap = cv2.VideoCapture(path)
	ret, frame = cap.read()
	if not ret:
		print("[FAIL] Cannot load video\n")
		sys.exit(1)
	cap.release()
	cv2.destroyAllWindows()
	print("[PASS] Video Loader\n")
	return 

if __name__ == "__main__":

	print("______ Overhead Detection ______\n")
	parser = argparse.ArgumentParser(description="Overhead Detection")

	parser.add_argument('-i','--input', default = '../data/challenge_clip.mkv', help='Path to source video')
	parser.add_argument('-o','--output', default = '../data/result_clip.mp4', help='Output file path and name')  # .mkv
	parser.add_argument('-f','--frames', default = 1, help='Number of frames to use for background')
	parser.add_argument('-w','--width', default = 450, help='Resize width keeping aspect ratio')
	parser.add_argument('-p','--rho', default = 0.2, help='Background model weight,i.e. bg(t) = p*I(t) + (1-p)*bg(t-1); range(0,1)')
	parser.add_argument('-t','--th', default = 20, help='Threshold for blob detection; range(0,255)')
	parser.add_argument('-m','--minb', default = 1000, help='Minimum blob size for detection; range(0, w*h)')

	args = parser.parse_args()

	testVideoLoad(args.input)

	main(args.input, args.output, args.frames, args.width, args.minb, args.rho, args.th)