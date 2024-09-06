import math
from scipy.special import erfc, erfcinv

# right now, only handles Base+Increment, probably fine simplification for most things
class BasicTimeControl:
    base = 15 # in minutes
    increment = 10 # in seconds
    def __init__(self, base, increment=0):
        self.base = base
        self.increment = increment

class Player:
    name = "Anonymous"
    country = "FID"
    rating_std = 0
    rating_blitz = 0
    rating_rapid = 0
    def __init__(self, name, rating = 0, rating_rapid = 0, rating_blitz = 0, country = "FID"):
        self.name = name
        self.country = country
        self.rating_std = rating
        self.rating_blitz = rating_blitz
        self.rating_rapid = rating_rapid
        if rating_blitz == 0:
            self.rating_blitz = rating
        if rating_rapid == 0:
            self.rating_rapid = rating
    def getRating(self, time): # Based on https://handbook.fide.com/chapter/B02RBRegulations2022
        if time.base + time.increment > 60:
            return self.rating_std
        elif time.base + time.increment > 10:
            return self.rating_rapid
        else:
            return self.rating_blitz
        
# calc based on https://wismuth.com/elo/calculator.html
        
FIDE_STDDEV = 2000 / 7

def eloNormal(eloDiff):
    return erfc(-eloDiff / (FIDE_STDDEV * math.sqrt(2))) / 2

def invEloNormal(p):
    return -(FIDE_STDDEV * math.sqrt(2)) * erfcinv(p * 2)

def addElo(diff, delta):
    if diff * delta >= 0:
        return diff + delta
    else:
        if diff > 0:
            return -invEloNormal(2 * eloNormal(-diff) - eloNormal(-diff + delta))
        else:
            return invEloNormal(2 * eloNormal(diff) - eloNormal(diff - delta))

def shiftedDiffs(white, black):
    diff = white - black
    ave = (white + black) / 2
    eloPerPawn = math.exp(ave/1020)*26.58903951991242
    c1 = eloPerPawn * 0.1 # win odds
    c2 = eloPerPawn * 0.6 # win+draw odds
    return (addElo(diff, c1-c2), addElo(diff, c1+c2))

def eloDiff(white, black):
    diff = white - black
    diffs = shiftedDiffs(white, black)
    if (diffs[0] > 0):
        return -invEloNormal((eloNormal(-diffs[0]) + eloNormal(-diffs[1]))/2)
    else:
        return invEloNormal((eloNormal(diffs[0]) + eloNormal(diffs[1]))/2)

def simGame(white, black, time):
