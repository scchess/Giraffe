import os
import shutil
import subprocess

os.environ["OMP_NUM_THREADS"] = "14"

def getLastIterationDone(filename):
	if not os.path.isfile(filename):
		return 0

	f = open(filename, 'r')

	last = 0
	for line in f:
		last = int(line.split(',')[0].strip())

	return last

startIteration = getLastIterationDone('testing.log') + 1

print("Starting on iteration " + str(startIteration) + " or later")

trainingLog = open('training.log', 'r')
testingLog = open('testing.log', 'a')

for line in trainingLog:
	splitLine = line.split(' ')
	iteration = int(splitLine[0].strip())
	evalFile = splitLine[1].strip()
	time = splitLine[2].strip()

	if iteration < startIteration:
		continue

	print("Testing " + evalFile)
	shutil.copyfile(evalFile, 'eval.t7')
	score = int(subprocess.check_output(["stsrun", "training/sts.epd", ".", "10000000000", "0.1", "0"]).strip())
	csvRow = str(iteration) + ", " + time + ", " + str(score)
	print(csvRow)
	testingLog.write(csvRow + "\n")
	testingLog.flush()
