import numpy as np


def B(kappa, tau):
    return (1 - np.exp(-kappa * tau)) / kappa


def A(kappa, mu, sigma, tau):
    B_val = B(kappa, tau)
    kappa2 = kappa * kappa
    sigma2 = sigma * sigma
    term1 = (B_val - tau) * (kappa2*mu - sigma2/2) / kappa2
    term2 = (sigma2 * B_val**2) / (4 * kappa)
    return np.exp(term1 - term2)


def current_bond_price(curr_short_rate, curr_z_prices):
    for i in range(0, l_scenarios):
        tau = (i + 1) * dt
        curr_z_prices[i] = np.exp(-B(kappa, tau) * curr_short_rate) * A(kappa, mu, sigma, tau)


def vasicek_model(n_scenarios, l_scenarios, scenarios_tau_length, r0, kappa, mu, dt, sigma):
    short_rates = np.zeros((n_scenarios, l_scenarios + 1))  # Creating an array for short rates
    z_prices = np.zeros((n_scenarios, l_scenarios + 1, scenarios_tau_length))  # Creating an array for zero coupon prices

    for n in range(0, n_scenarios):
        # Dealing with tau=0: The same initial value for all scenarios
        short_rates[n, 0] = r0
        curr_short_rate = short_rates[n, 0]
        curr_z_prices = np.zeros(scenarios_tau_length)
        current_bond_price(curr_short_rate, curr_z_prices)
        for i in range(0, scenarios_tau_length):
            z_prices[n, 0, i] = curr_z_prices[i]
        for j in range(1, l_scenarios + 1):
            short_rates[n,  j] = kappa*(mu - short_rates[n, j-1])*dt + sigma * np.random.randn() * np.sqrt(dt) + short_rates[n, j-1]
            curr_short_rate = short_rates[n, j]
            curr_z_prices = np.zeros(scenarios_tau_length)
            current_bond_price(curr_short_rate, curr_z_prices)
            for i in range(0, scenarios_tau_length):
                z_prices[n, j, i] = curr_z_prices[i]
    return z_prices

def irs_exposure(z_prices, l_scenarios, n_scenarios, SwapRate, alpha):
    IRS_EE = np.zeros(l_scenarios + 1, dtype=float, order='C')  # array for the Expected Exposure values
    IRS_PFE = np.zeros(l_scenarios + 1, dtype=float, order='C')  # array for the PFEk values
    IRS_MtM = np.zeros((n_scenarios, l_scenarios + 1))  # array for MtM values
    SwapFxdLegs = np.zeros((n_scenarios, l_scenarios + 1))
    SwapFltLegs = np.zeros((n_scenarios, l_scenarios + 1))
    for n in range(0, n_scenarios):
        for j in range(0, l_scenarios):
            for i in range(0, scenarios_tau_length - j):
                SwapFxdLegs[n, j] += z_prices[n, j, i]
            SwapFxdLegs[n, j] = dt * SwapRate * SwapFxdLegs[n, j]
            SwapFltLegs[n, j] = 1 - z_prices[n, j, l_scenarios - j - 1]
            IRS_MtM[n, j] = SwapFxdLegs[n, j] - SwapFltLegs[n, j]
        SwapFxdLegs[n, l_scenarios] = 0
        SwapFltLegs[n, l_scenarios] = 0
        IRS_MtM[n, l_scenarios] = 0
    k = int(n_scenarios * alpha)
    print(alpha, k)
    for j in range(0, l_scenarios + 1):
        MtMCurJ = np.zeros(n_scenarios)
        IRS_EE[j] = 0.0
        for n in range(0, n_scenarios):
            MtMCurJ[n] = IRS_MtM[n, j]
            if MtMCurJ[n] > 0:
                IRS_EE[j] = IRS_EE[j] + MtMCurJ[n]
        IRS_EE[j] = IRS_EE[j] / n_scenarios
        MtMCurJsrd = np.sort(MtMCurJ)
        IRS_PFE[j] = MtMCurJsrd[k - 1]
    return IRS_MtM, IRS_EE, IRS_PFE


if __name__ == '__main__':
    mu = 0.05
    kappa = 0.1
    sigma = 0.01
    r0 = 0.05
    n_scenarios = 250
    dt = 0.25
    T = 10
    l_scenarios = int(T / dt)
    tau_T = 10
    dtau = 0.25
    SwapRate = 0.05
    scenarios_tau_length = int(tau_T / dtau)
    alpha = 0.95

    z_prices_ = vasicek_model(n_scenarios, l_scenarios, scenarios_tau_length, r0, kappa, mu, dt, sigma)
    MtM, EE, PFE = irs_exposure(z_prices_, l_scenarios, n_scenarios, SwapRate, alpha)