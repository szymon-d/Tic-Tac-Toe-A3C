from utils.network import A3C_Network
from settings import settings


global_network = A3C_Network(**settings)
global_network.load(traning_mode=False)


