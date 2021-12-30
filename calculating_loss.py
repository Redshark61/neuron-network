import math

# High confidence of output
softmaxOutput = [0.7, 0.1, 0.2]
targetOutput = [1, 0, 0]

loss = - (math.log(softmaxOutput[0])*targetOutput[0] +
          math.log(softmaxOutput[1])*targetOutput[1] +
          math.log(softmaxOutput[2])*targetOutput[2])
# Low loss bcz confidence is high
print(loss)
print(- (math.log(softmaxOutput[0])))

# Low confidence of output
softmaxOutput = [0.5, 0.3, 0.2]
# High loss bcz confidence is low
print(- (math.log(softmaxOutput[0])))
