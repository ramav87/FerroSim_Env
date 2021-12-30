#defects

class LatticeDefect():
    def __init__(self, Ef=(0, 0), loc=[0, 0], idx=0):
        self.Ef = Ef
        self.loc = loc
        self.idx = idx
        self.loc_history = []
        self.loc_history.append(loc)

    def __repr__(self):
        return ("Lattice Defect #{}, Located at {}, \
    with Field Strength {}".format(self.idx, self.loc, self.Ef))

    def return_idx(self):
        return self.idx

    def update_location(self, loc):
        # Update defect location and update defect location history
        self.loc = loc
        self.loc_history.append(loc)