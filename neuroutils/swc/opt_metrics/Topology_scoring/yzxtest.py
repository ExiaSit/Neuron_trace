#
#
# gt = []
# pred = []
# with open(r"C:\Users\Administrator\Desktop\graph\vancouver.graph") as f:
#     _switch = False
#     while True:
#         line = f.readline()
#         if line == "\n":
#             _switch = True
#         if line != "" and not _switch:
#             line = line.strip("\n")
#             line = line + " 1.0\n"
#             gt.append(line)
#         elif line != "" and _switch:
#             gt.append(line)
#         else:
#             break
# with open(r"C:\Users\Administrator\Desktop\graph\3d_vancouver.graph", "w") as f:
#     f.writelines(gt)
#
# with open(r"C:\Users\Administrator\Desktop\graph\vancouver.graph") as f:
#     _switch = False
#     while True:
#         line = f.readline()
#         if line == "\n":
#             _switch = True
#         if line != "" and not _switch:
#             line = line.strip("\n")
#             line = line + " 1.0\n"
#             gt.append(line)
#         elif line != "" and _switch:
#             gt.append(line)
#         else:
#             break
# with open(r"C:\Users\Administrator\Desktop\graph\3d_vancouver.graph", "w") as f:
#     f.writelines(gt)

import numpy as np
