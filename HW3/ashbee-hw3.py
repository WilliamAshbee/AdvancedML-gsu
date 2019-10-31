



sequences = []
with open("HW4.fas",'r') as infile:
    for line in infile.readlines():
#        print(line)
        sequence = True
        for char in line.strip():
            if not (char == 'A' or  char == 'C' or char == 'G' or char == 'T'):
                sequence = False
        if sequence:
            print(line)
            sequences.append(line)
            assert line == sequences[-1]

print(sequences[-1])