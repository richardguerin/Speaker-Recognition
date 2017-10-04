# .wav file directories
trainDir$ = "/home/ricky/Projects/SpeakerRec/Training"
testDir$ = "/home/ricky/Projects/SpeakerRec/Testing"

# .csv output files
trainOut$ = "/home/ricky/Projects/SpeakerRec/trainData.csv"
testOut$ = "/home/ricky/Projects/SpeakerRec/testData.csv"
deleteFile: trainOut$ 
deleteFile: testOut$ 

# Append headers to .csv
appendFileLine: trainOut$, "meanF1,meanF2,meanF3,meanF4,meanF5,dispF5F1,dispF4F3,dispF5F3,dispF5F4,meanf0,meanJitter,meanShimmer,dx"
appendFileLine: testOut$, "meanF1,meanF2,meanF3,meanF4,meanF5,dispF5F1,dispF4F3,dispF5F3,dispF5F4,meanf0,meanJitter,meanShimmer,dx"

# Append data to trainData.csv
strings = Create Strings as file list: "list", trainDir$ + "/*.wav"
numberOfFiles = Get number of strings
for i to numberOfFiles
	selectObject: strings

	# Get File name
	fileName$ = Get string: i
	fileIntro$ = replace$ (fileName$, ".wav", "", 1)
	appendInfoLine: string$ (i) + ": " + fileIntro$

	# Get Formant object
	Read from file: trainDir$ +"/"+ fileName$
	selectObject: "Sound " + fileIntro$
	To Formant (burg): 0, 5, 5500, 0.025, 50

	# Extract each mean formant
	meanfOne$ = Get mean: 1, 0, 0, "Hertz"
	meanfTwo$ = Get mean: 2, 0, 0, "Hertz"
	meanfThree$ = Get mean: 3, 0, 0, "Hertz"
	meanfFour$ = Get mean: 4, 0, 0, "Hertz"
	meanfFive$ = Get mean: 5, 0, 0, "Hertz"

	# Append mean F1 to F5
	appendFile: trainOut$, number (meanfOne$), ",", number (meanfTwo$), ",", number (meanfThree$), ","
	appendFile: trainOut$, number (meanfFour$), ",", number (meanfFive$), ","

	# Append formant dispersions
	appendFile: trainOut$, number (meanfFive$) - number (meanfOne$), ","
	appendFile: trainOut$, number (meanfFour$) - number (meanfThree$), ","
	appendFile: trainOut$, number (meanfFive$) - number (meanfThree$), ","
	appendFile: trainOut$, number (meanfFive$) - number (meanfFour$), ","

	# Get and append F0
	selectObject: "Sound " + fileIntro$
	To Pitch (cc): 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600
	meanfZero$ = Get mean: 0, 0, "Hertz"
	appendFile: trainOut$, number (meanfZero$), ","

	# Get voice report
	selectObject: "Sound " + fileIntro$
	plusObject: "Pitch " + fileIntro$
	To PointProcess (cc)

	selectObject: "Sound " + fileIntro$
	plusObject: "PointProcess " + fileIntro$ + "_" + fileIntro$
	plusObject: "Pitch " + fileIntro$
	report$ = Voice report: 0, 0, 75, 600, 1.3, 1.6, 0.03, 0.45

	# Append Jitter and Shimmer
	j = extractNumber (report$, "Jitter (local): ")
	s = extractNumber (report$, "Shimmer (local): ")

	appendFile: trainOut$, j, ",", s, ","

	# Append 1 if Ricky's voice, 0 if not
	if startsWith (fileIntro$, "Italian")
		appendFileLine: trainOut$, "1"
	elif startsWith (fileIntro$, "MoreRicky")
		appendFileLine: trainOut$, "1"
	elif startsWith (fileIntro$, "MORE")
		appendFileLine: trainOut$, "1"
	elif startsWith (fileIntro$, "Ricky")
		appendFileLine: trainOut$, "1"
	else
		appendFileLine: trainOut$, "0"
	endif

	removeObject: "Sound " + fileIntro$
	removeObject: "Pitch " + fileIntro$
	removeObject: "PointProcess " + fileIntro$ + "_" + fileIntro$
	removeObject: "Formant " + fileIntro$
endfor


# Append data to testData.csv
strings = Create Strings as file list: "list", testDir$ + "/*.wav"
numberOfFiles = Get number of strings
for i to numberOfFiles
	selectObject: strings

	# Get File name
	fileName$ = Get string: i
	fileIntro$ = replace$ (fileName$, ".wav", "", 1)
	appendInfoLine: string$ (i) + ": " + fileIntro$

	# Get Formant object
	Read from file: testDir$ +"/"+ fileName$
	selectObject: "Sound " + fileIntro$
	To Formant (burg): 0, 5, 5500, 0.025, 50

	# Extract each mean formant
	meanfOne$ = Get mean: 1, 0, 0, "Hertz"
	meanfTwo$ = Get mean: 2, 0, 0, "Hertz"
	meanfThree$ = Get mean: 3, 0, 0, "Hertz"
	meanfFour$ = Get mean: 4, 0, 0, "Hertz"
	meanfFive$ = Get mean: 5, 0, 0, "Hertz"

	# Append mean F1 to F5
	appendFile: testOut$, number (meanfOne$), ",", number (meanfTwo$), ",", number (meanfThree$), ","
	appendFile: testOut$, number (meanfFour$), ",", number (meanfFive$), ","

	# Append formant dispersions
	appendFile: testOut$, number (meanfFive$) - number (meanfOne$), ","
	appendFile: testOut$, number (meanfFour$) - number (meanfThree$), ","
	appendFile: testOut$, number (meanfFive$) - number (meanfThree$), ","
	appendFile: testOut$, number (meanfFive$) - number (meanfFour$), ","

	# Get and append F0
	selectObject: "Sound " + fileIntro$
	To Pitch (cc): 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600
	meanfZero$ = Get mean: 0, 0, "Hertz"
	appendFile: testOut$, number (meanfZero$), ","

	# Get voice report
	selectObject: "Sound " + fileIntro$
	plusObject: "Pitch " + fileIntro$
	To PointProcess (cc)

	selectObject: "Sound " + fileIntro$
	plusObject: "PointProcess " + fileIntro$ + "_" + fileIntro$
	plusObject: "Pitch " + fileIntro$
	report$ = Voice report: 0, 0, 75, 600, 1.3, 1.6, 0.03, 0.45

	# Append Jitter and Shimmer
	j = extractNumber (report$, "Jitter (local): ")
	s = extractNumber (report$, "Shimmer (local): ")

	appendFile: testOut$, j, ",", s, ","

	# Append 1 if Ricky's voice, 0 if not
	if startsWith (fileIntro$, "MoreRicky")
		appendFileLine: testOut$, "1"
	elif startsWith (fileIntro$, "MORE")
		appendFileLine: testOut$, "1"
	elif startsWith (fileIntro$, "Ricky")
		appendFileLine: testOut$, "1"
	else
		appendFileLine: testOut$, "0"
	endif

	removeObject: "Sound " + fileIntro$
	removeObject: "Pitch " + fileIntro$
	removeObject: "PointProcess " + fileIntro$ + "_" + fileIntro$
	removeObject: "Formant " + fileIntro$
endfor
