import glob
filenames = glob.glob('./*.dat')
filedata = {filename: open(filename, 'r') for filename in filenames}

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

fpow = []
delta_bin = {}

for file in filedata.values():
    fpow.append(float(remove_prefix(file.readline().splitlines()[0],"<f> = ")))
    for line in file.read().splitlines():
        kv = [int(i) for i in line.split(' ')]
        if kv[0] in delta_bin.keys():
            vo = delta_bin[kv[0]]
            vo.append(kv[1])
            delta_bin.update({kv[0]: vo})
        else:
            newList = []
            newList.append(kv[1])
            delta_bin.update({kv[0]: newList})

for file in filedata.values():
    file.close()

xdata = list(range(0,150+1))

ydata = {}

for i in xdata:
    if i in delta_bin.keys():
        ydata.update({i:delta_bin[i]})
    else:
        ydata.update({i:[0]})

ymean = []
ymin = []
ymax = []
for i in xdata:
    yd = [ele / 1000.0 for ele in ydata[i]]
    mean = sum(yd)/len(yd)
    var = sum([((x - mean) ** 2) for x in yd]) / len(yd)
    std = var ** 0.5
    ymean.append(mean)
    ymin.append(mean-std/2)
    ymax.append(mean+std/2)

out = open("data_set.out","w")
for i in xdata:
    out.write(str(i)+" "+str(ymean[i])+" "+str(ymin[i])+" "+str(ymax[i])+"\n")
    
out.close()
