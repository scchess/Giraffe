import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILENAME = '/var/www/html/plot.png'

plt.rcParams["figure.figsize"] = [15, 8]

if len(sys.argv) < 2:
	print("Usage: " + sys.argv[0] + " <file1> <file2> ...")
	sys.exit()

for i in xrange(1, len(sys.argv)):
	filename = sys.argv[i]
	data = []

	with open(filename, 'rb') as csvFile:
		reader = csv.reader(csvFile, delimiter=' ')
		for row in reader:
			time = row[2].strip()
			score = row[3].strip()

			data.append([float(time), float(score)])

	if len(data) > 0:
		dataArray = np.asarray(data, dtype=np.float32)
		plt.plot(dataArray[:, 0], dataArray[:, 1], label=filename)

plt.xlabel('Time (s)')
plt.ylabel('STS score (0.1s)')
plt.grid(True)
plt.legend(loc='best')
plt.savefig(OUTPUT_FILENAME)

print("Saved to " + OUTPUT_FILENAME)
