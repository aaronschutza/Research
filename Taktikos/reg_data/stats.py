import glob
filenames = glob.glob('./*.dat')
filenames.sort()
filedata = {filename: open(filename, 'r') for filename in filenames}

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

fpow = [0.0]
xgrid = range(21)

for file in filedata.values():
    fpow.append(float(remove_prefix(file.readline().splitlines()[0],"<f> = ")))

for file in filedata.values():
    file.close()

out = open("data_set.out","w")
out2 = open("data_set_2.out","w")

for i in xgrid:
    out.write(str(i*0.05)+" "+str(fpow[i]/(fpow[i]+fpow[20-i]))+"\n")
    out2.write(str(i*0.05)+" "+str(fpow[i])+"\n")

out.close()
out2.close()
