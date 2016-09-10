import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILENAME = '/var/www/html/plot.png'

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " <file1> <file2> ...")
	sys.exit()

for i in xrange(1, len(sys.argv)):
	filename = sys.argv[i]
	data = []

	with open(filename, 'rb') as csvFile:
		reader = csv.reader(csvFile, delimiter=',')
		for row in reader:
			if (len(row) == 3):
				time = row[1].strip()
				score = row[2].strip()
			else:
				# old format without iteration count
				time = row[0].strip()
				score = row[1].strip()

			data.append([float(time), float(score)])

	dataArray = np.asarray(data, dtype=np.float32)

	plt.plot(dataArray[:, 0], dataArray[:, 1], label=filename)

plt.xlabel('Time (s)')
plt.ylabel('STS score (0.1s)')
plt.grid(True)
plt.legend(loc='best')
plt.savefig(OUTPUT_FILENAME)

print("Saved to " + OUTPUT_FILENAME)