import kwant
import numpy as np


def build_system(sites,builder="kwant"):
    syst=kwant.Builder()
    return syst