tru=[[0,1,1,1,1,1],[0,2,2,0,1,1,1,1,0,0],[2,2,2,2,0,0,0,0,1,1,1,1,0,0,0,1,1,1]]
startss=[]
endss = []
labels = []
for i in tru:
    starts=[]
    ends=[]
    label=[]
    before = 0
    for j in range(len(i)):
        if i[j]!=before and i[j]!=0:
            starts.append(j)
        if j!=len(i)-1:
            if i[j]!=i[j+1] and i[j]!=0:
                label.append(i[j])
                ends.append(j)
        else:
            if i[j] == i[j-1] and i[j]!=0:
                label.append(i[j])
                ends.append(j)

        before = i[j]
    startss.append(starts)
    endss.append(ends)
    labels.append(label)
    print(starts)
    print(ends)
    print(label)
    print("-----")

    for k in range(len(starts)):
        start_positions = [0] * (len(i))
        for jj in range(len(start_positions)):
            if starts[k]<=jj<=ends[k]:
                start_positions[jj]=label[k]
        print(start_positions)

