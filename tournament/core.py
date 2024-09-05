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
