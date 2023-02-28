
import numpy as np

myStr = "model"
myVars = locals()
for i in range(0,10):
    myVars.__setitem__(myStr + str(i), np.array([1, 0, 0, 1, 0, 0, 1])*i)

i=5
print(model5)
myVars.__setitem__(myStr + str(i), myVars.__getitem__(myStr + str(i)) + 2)
# print(model1)
print(model5)  