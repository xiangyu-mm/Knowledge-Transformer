num = 0
with open('data.txt','r') as file:
	for line in file.readlines():
		for i in line:
			if i == 'Q' or i == 'u':
				num += 1
		print(num)
	print(num)