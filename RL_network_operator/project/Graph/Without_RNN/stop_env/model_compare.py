from Util import *
import pickle
filehandler = open("trival_environment","rb")
environment = pickle.load(filehandler)
filehandler.close()
print(reject_when_full(environment))

filehandler = open("trival_environment","rb")
environment = pickle.load(filehandler)
filehandler.close()
print(totally_random(environment))