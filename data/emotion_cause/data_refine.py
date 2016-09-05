import numpy as np
f = open('No Cause.txt','r')
fw = open('No Cause refine.txt','w')
count = np.zeros(7)
for line in f:
	x = line.split('<\\')[1].split('>')[0]
	temp =  line.split('<\\')[0].split('>')[1]
	if x == 'happy':
		fw.write('0\t')
		count[0] = count[0] + 1
	elif x=='sad':
		fw.write('1\t')
		count[1] = count[1] + 1
	elif x=='anger':
		fw.write('2\t')
		count[2] = count[2] + 1
	elif x=='fear':
		fw.write('3\t')
		count[3] = count[3] + 1
	elif x=='surprise':
		fw.write('4\t')
		count[4] = count[4] + 1
	elif x=='disgust':
		fw.write('5\t')
		count[5] = count[5] + 1
	elif x=='shame':
		fw.write('6\t')
		count[6] = count[6] + 1
	
	fw.write(temp)
	fw.write('\n')
print count
f.close()