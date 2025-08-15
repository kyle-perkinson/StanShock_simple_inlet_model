import cantera as ct

class FlowState:
    def __init__(self, composition: str = "air"):
        if composition == "air":
            self.mech = "data/mechanisms/N2O2HeAr.yaml"
            self.X = "O2:0.21 N2:0.79"
        else:
            self.mech = composition
            self.X = composition
        self.gas = ct.Solution(self.mech)
        self.k = None
        self.M = None
        self.theta = None
        self.P = None
        self.T = None
        self.rho = None

    def set_state(self, T: float, P: float, M: float, theta: float):
        self.gas.TPX = T, P, self.X
        self.k = self.gas.cp_mass / self.gas.cv_mass
        self.M = M
        self.theta = theta
        self.P = P
        self.T = T
        self.rho = self.gas.density

    def clone(self):
        fs = FlowState(self.mech if self.X != "O2:0.21 N2:0.79" else "air")
        fs.set_state(self.T, self.P, self.M, self.theta)
        return fs