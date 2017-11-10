DATA_DIR = "/home/naanu/Music/dataForTest/amazonPosNegDataNew.txt"
DATA_wr = "/home/naanu/Music/dataForTest/amazonPosNegDataFinal.txt"
"""
#fileWrite = open(DATA_wr, "a")
count = 0
with open(DATA_DIR, 'r') as textFile:
  for line in textFile:
	line = line+" "
	label, sentence = line.split("\t",1)
	print(label)
	print(sentence)
	sentence = label + "\t" + sentence.strip()
#	print("\n"+sentence+"\n")# label, sentence = sentence.split("\t", 1)
#	print("\n"+label+"\n")
#	fileWrite.write(sentence+"\n")
#	count = count + 1
#	print(count)
	label, sentence = sentence.split("\t",1)
	print("after split:")
	print(label+"\n")
	print(sentence)
#	break

#fileWrite.close()
print("\nResult for Counting")

"""

fileWrite = open(DATA_wr, "a")
count = 0
with open(DATA_DIR, 'r') as tfile:
	for line in tfile:
		line = line.replace("\t","")
		line = line.strip()
		if line != " ":
			print(line)
			count = count + 1
			#if count == 3913:
			#	print(line)
			#	break
			fileWrite.write(line)
			print(count)
