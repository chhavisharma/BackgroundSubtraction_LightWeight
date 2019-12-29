# LRMtest
Assignement


# Installation 
Use conda to create a new environment 
`
conda create -n lrm python=3.6 anaconda
`


# Run
cd LRMtest/data
python main.py

`
	parser.add_argument('-i','--input', default = '../data/challenge_clip.mkv', help='Path to source video')
	parser.add_argument('-o','--output', default = '../data/result_clip.mp4', help='Output file path and name')  # .mkv
	parser.add_argument('-f','--frames', default = 1, help='Number of frames to use for background')
	parser.add_argument('-w','--width', default = 450, help='Resize width keeping aspect ratio')
	parser.add_argument('-p','--rho', default = 0.2, help='Background model weight,i.e. bg(t) = p*I(t) + (1-p)*bg(t-1); range(0,1)')
	parser.add_argument('-t','--th', default = 20, help='Threshold for blob detection; range(0,255)')
	parser.add_argument('-m','--minb', default = 1000, help='Minimum blob size for detection; range(0, w*h)')

`
