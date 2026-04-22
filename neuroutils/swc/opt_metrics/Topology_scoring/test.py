import numpy as np

# C:\\Users\\Administrator\\Desktop\\swc\\app2\\gt 4332_6760_6296.v3draw_manual.swc
# C:\\Users\\Administrator\\Desktop\\swc\\app2\\app2 4332_6760_6296.v3draw_app2.swc

# C:\\Users\\Administrator\\Desktop\\swc\\match\\new  4332_6760_6296.swc
# C:\\Users\\Administrator\\Desktop\\swc\\match\\raw  4332_6760_6296.swc

f = open('C:\\Users\\Administrator\\Desktop\\swc\\app2\\gt\\4332_6760_6296.v3draw_manual.swc', 'r', encoding='utf-8')
f.seek(0)
a = f.readline()
b = f.readline()  # 去掉前面那几行
e = f.readline()
c = ""  # 坐标
d = ""  # 连接
swc = []
for line in f:
    line = line.strip()
    lsp = line.split(' ')
    n, _type, x, y, z, r, parent = lsp
    swc.append([int(n), int(_type), float(x), float(y), float(z), float(r), int(parent)])


def reindex_swc(swc: list):
    swc_arr = np.array(swc)
    newidx = np.arange(0, len(swc_arr), 1)  # 长度 步距为1
    oldpid = swc_arr[:, 6]
    newpid = []
    mapdict = {}  # 映射关系
    for i, node in enumerate(swc_arr):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标。
        print(node)
        id = node[0]  # n
        mapdict[id] = i  # 索引序列
    for pid in oldpid:
        if pid == -1:
            newpid.append(-1)
        else:
            newpid.append(mapdict[pid])
    newswc = swc_arr.copy()
    newswc[:, 0] = newidx
    newswc[:, 6] = newpid
    return list(newswc)


swc = reindex_swc(swc)
# print(swc[:100])
# input()
swc = np.asarray(swc)
swc = swc[swc[:, 0].argsort()]
idlist = np.array(swc)[:, 0]
for node in swc:
    n, _type, x, y, z, r, parent = node
    # c = str(x) + ' ' + str(y) + ' ' + str(z)
    c = " ".join([str(x), str(y), str(z)])
    with open('C:\\Users\\Administrator\\Desktop\\swc\\match\\raw\\shang.swc', "a") as f:
        f.writelines(c + '\n')

    if parent in idlist:
        d = "{0:.0f} {1:.0f}".format(n, parent)
        with open('C:\\Users\\Administrator\\Desktop\\swc\\match\\raw\\xia.swc', "a") as f:
            f.writelines(d + '\n')
        d = "{0:.0f} {1:.0f}".format(parent, n)
        with open('C:\\Users\\Administrator\\Desktop\\swc\\match\\raw\\xia.swc', "a") as f:
            f.writelines(d + '\n')

with open('C:\\Users\\Administrator\\Desktop\\swc\\match\\raw\\shang.swc', "a") as f:
    f.writelines('\n')

f1 = open('C:\\Users\\Administrator\\Desktop\\swc\\match\\raw\\xia.swc', "r", encoding='utf-8')
for line in f1:
    with open('C:\\Users\\Administrator\\Desktop\\swc\\match\\raw\\shang.swc', "a") as f:
        f.writelines(line)
