"""
This module provides functions for fitting experimental data to cooperative binding models,
including the isodesmic and cooperative polymerization models.
It also includes utilities for data loading, manipulation, and visualization of fitting results.
"""

from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt
import lmfit as lf
import base_utils.plotlib as pl


### deplicated decorator
import warnings


def deprecated(reason: str):
    def decorator(func):
        def new_func(*args, **kwargs):
            warnings.warn(
                f"Function {func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return new_func

    return decorator


###


def isodesmic(X: np.number | npt.NDArray[np.number], K: np.number | float | int):
    """
    Calculates the fraction of monomer in an isodesmic polymerization model.
    Args:
        X (np.number | npt.NDArray[np.number]): Concentration of monomer (e.g., UV/Vis absorbance).
        K (np.number | float | int): Equilibrium constant.
    Returns:
        np.number | npt.NDArray[np.number]: Fraction of monomer.
    """
    B = K * X
    return (2 * B + 1 - (4 * B + 1) ** 0.5) / (2 * B)


def real_cbrt(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.sign(x) * np.abs(x) ** (1 / 3)


def cubic(
    b: npt.NDArray[np.float64], c: npt.NDArray[np.float64], d: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    p = (3 * c - b**2) / 3
    q = (2 * b**3 - 9 * b * c + 27 * d) / 27
    delta = (q / 2) ** 2 + (p / 3) ** 3

    # 出力配列を準備
    x = np.empty_like(b, dtype=np.float64)

    mask1 = delta >= 0
    if np.any(mask1):
        u = real_cbrt(-q[mask1] / 2 + np.sqrt(delta[mask1]))
        v = real_cbrt(-q[mask1] / 2 - np.sqrt(delta[mask1]))
        y = u + v
        x[mask1] = (y - b[mask1] / 3).real

    mask2 = ~mask1
    if np.any(mask2):
        r = 2 * np.sqrt(-p[mask2] / 3)
        arg = (-q[mask2] / 2) / np.sqrt(-((p[mask2] / 3) ** 3))
        arg = np.clip(arg, -1.0, 1.0)  # 丸め誤差対策
        theta = np.arccos(arg)
        y1 = r * np.cos(theta / 3)
        y2 = r * np.cos(theta / 3 - 2 * np.pi / 3)
        y3 = r * np.cos(theta / 3 - 4 * np.pi / 3)
        y = np.minimum(np.minimum(y1, y2), y3)
        x[mask2] = (y - b[mask2] / 3).real

    return x


@deprecated("Use cooperative_stabilized instead")
def cooperative(
    Conc: float | npt.NDArray[np.number],
    K: float | np.number,
    sigma: float | np.number,
    scaler: float | np.number = 1,
) -> float | npt.NDArray[np.number]:
    """
    Calculates the rate of supramolecular polymer formation based on a cooperative binding model.
    Args:
        Conc (float | npt.NDArray[np.number]): Total concentration of the substrate.
        K (float | np.number): Equilibrium constant.
        sigma (float | np.number): Cooperativity factor.
        scaler (float | np.number): Scaling factor for the output.
    Returns:
        float | npt.NDArray[np.number]: Rate of supramolecular polymer formation.
    """
    scaled_conc = K * Conc

    b = -(2 + scaled_conc / (1 - sigma)) + 0j
    c = 1 + sigma / (1 - sigma) + 2 * scaled_conc / (1 - sigma) + 0j
    d = -scaled_conc / (1 - sigma) + 0j
    # x^3 + bx^2 + cx + d = 0
    # solve x with Cardano's formula

    ans = cubic(b, c, d)
    # ans =[[x for x in a if x >= -0.001 and x <= con] for a,con in zip(ans,scaled_conc)]

    # for i in range(len(ans)):
    #     if len(ans[i]) == 0:
    #         ans[i] = [scaled_conc[i]]
    # mono_conc = [x[0] for x in ans]
    mono_conc = ans
    mono_conc = np.array(mono_conc)
    mono_conc = mono_conc / K
    # ans = 1- mono_conc/Conc
    ans = 1 - np.divide(mono_conc, Conc, out=np.ones_like(mono_conc), where=Conc != 0)
    ans = [x if not np.isnan(x) else 0 for x in ans]
    ans = np.array(ans)

    return ans * scaler


def solve_cubic_vectorized(
    a, b, c, d, x_low, x_high, max_iter=60
) -> npt.NDArray[np.float64]:
    """
    Solve a x^3 + b x^2 + c x + d = 0 on the interval [x_low, x_high]
    using fully vectorized bisection.
    Args:
        a, b, c, d: Coefficients of the cubic equation.
        x_low, x_high: Bounds for the root search.
        max_iter: Maximum number of iterations for bisection.
    Returns:
        npt.NDArray[np.float64]: Approximated roots within the specified bounds.

    Raises:
        ValueError: If the root is not bracketed for some elements.
        RuntimeError: If the bisection does not converge within the maximum number of iterations.
    """

    # Shape unification
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)

    # Initial bounds
    xl: npt.NDArray[np.float64] = np.asarray(x_low, dtype=np.float64)
    xr: npt.NDArray[np.float64] = np.asarray(x_high, dtype=np.float64)

    # Evaluate cubic
    def f(x):
        return a * x**3 + b * x**2 + c * x + d

    fl = f(xl)
    fr = f(xr)
    mask_valid = fl * fr <= 0
    if not np.all(mask_valid):
        raise ValueError("Root not bracketed for some elements.")

    # Bisection method loop
    for _ in range(max_iter):
        xm = 0.5 * (xl + xr)
        fm = f(xm)

        left_mask = fl * fm <= 0
        xr[left_mask] = xm[left_mask]
        xl[~left_mask] = xm[~left_mask]
        fl = f(xl)

    if not np.all(np.abs(xr - xl) < 1e-12):
        raise RuntimeError(
            "Bisection did not converge within the maximum number of iterations."
        )
    return 0.5 * (xl + xr)


def cooperative_stabilized(
    Conc: float | npt.NDArray[np.number],
    K: float | np.number,
    sigma: float | np.number,
    scaler: float | np.number = 1,
) -> float | npt.NDArray[np.number]:
    """
    Calculates the rate of supramolecular polymer formation based on a cooperative binding model.
    Args:
        Conc (float | npt.NDArray[np.number]): Total concentration of the substrate.
        K (float | np.number): Equilibrium constant.
        sigma (float | np.number): Cooperativity factor.
        scaler (float | np.number): Scaling factor for the output.
    Returns:
        float | npt.NDArray[np.number]: Rate of supramolecular polymer formation.
    """
    scaled_conc = K * Conc
    a = 1 - sigma
    b = -(2 * a + scaled_conc)
    c = 1 + 2 * scaled_conc
    d = -scaled_conc
    # x^3 + bx^2 + cx + d = 0
    # max concentration of monomer (scaled) is [0,1]
    # because of convergence condition
    x_low = np.zeros_like(scaled_conc)
    x_high = np.ones_like(scaled_conc)

    mono_conc = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
    mono_conc = mono_conc / K
    # ans = 1- mono_conc/Conc
    ans = 1 - np.divide(mono_conc, Conc, out=np.ones_like(mono_conc), where=Conc != 0)
    ans = [x if not np.isnan(x) else 0 for x in ans]
    ans = np.array(ans)

    return ans * scaler


def temp_cooperative(Temp, deltaH, deltaS, deltaHnuc, c_tot, scaler=1) -> np.ndarray:
    """
    Calculates the cooperative binding based on temperature-dependent parameters.
    Args:
        Temp: Temperature.
        deltaH: Enthalpy change.
        deltaS: Entropy change.
        deltaHnuc: Nucleation enthalpy change.
        c_tot: Total concentration.
        scaler: Scaling factor.
    Returns:
        np.ndarray: Cooperative binding values.
    """
    R = 8.314

    K = np.exp(-deltaH / (R * Temp) + deltaS / R)
    sigma = np.exp(deltaHnuc / (R * Temp))

    return cooperative(c_tot, K, sigma, scaler=scaler)


def model(
    params: lf.Parameters, x: np.ndarray, c_tot: float, scaler: float
) -> np.ndarray:
    """
    Defines the cooperative binding model for fitting.
    Args:
        params (lmfit.Parameters): Parameters for the model (deltaH, deltaS, deltaHnuc).
        x (np.ndarray): X values (e.g., temperature).
        c_tot (float): Total concentration.
        scaler (float): Scaling factor.
    Returns:
        np.ndarray: Calculated Y values from the model.
    """
    # unpack params
    deltaH = params["deltaH"]
    deltaS = params["deltaS"]
    deltaHnuc = params["deltaHnuc"]

    return temp_cooperative(x, deltaH, deltaS, deltaHnuc, c_tot, scaler=scaler)


def objective(params: lf.Parameters, data: list[pl.XYData]) -> np.ndarray:
    """
    Defines the objective function for fitting, calculating residuals between
    measured data and the model's predictions.
    Args:
        params (lmfit.Parameters): Parameters for the model.
        data (list[pl.XYData]): List of measured X-Y datasets.
    Returns:
        np.ndarray: Array of residuals.
    """

    nX = [len(d.X) for d in data]
    nX = np.sum(nX)

    residual = np.zeros(nX)
    index = 0

    for i, d in enumerate(data):
        res = d.Y - model(
            params, d.X, c_tot=int(d.Title) / 1000000, scaler=params["scaler" + str(i)]
        )
        residual[index : index + len(d.X)] = res
        index += len(d.X)
    # now flatten this to a 1D array, as minimize() needs
    return residual


def objective2(params: lf.Parameters, data: list[pl.XYData]) -> np.ndarray:
    """
    Defines an alternative objective function for fitting, including a linear background term.
    Calculates residuals between measured data and the model's predictions with background.
    Args:
        params (lmfit.Parameters): Parameters for the model, including background parameters.
        data (list[pl.XYData]): List of measured X-Y datasets.
    Returns:
        np.ndarray: Array of residuals.
    """

    nX = [len(d.X) for d in data]
    nX = np.sum(nX)

    residual = np.zeros(nX)
    index = 0

    for i, d in enumerate(data):
        res = (
            d.Y
            - model(
                params,
                d.X,
                c_tot=int(d.Title) / 1000000,
                scaler=params["scaler" + str(i)],
            )
            + params["background" + str(i)] * d.X
        )
        residual[index : index + len(d.X)] = res
        index += len(d.X)
    # now flatten this to a 1D array, as minimize() needs
    return residual


def main():
    """
    Main function for testing the cooperative binding model and plotting results.
    """
    X = np.linspace(0, 800, 10000)
    Y = temp_cooperative(X, -102000, -198, -20000, 0.000030, scaler=0.5)
    print(Y)
    X = X - 273.15
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(4, 3)
    ax.minorticks_on()
    ax.set_xlim(0, 140)
    ax.set_xlabel("X")

    ax.set_ylabel("Y")
    ax.set_title("Cooperative binding")
    ax.plot(X, Y)
    plt.show()


def fit(path: str = "UV", clip: int = 65):
    """
    Performs fitting of experimental data to the cooperative binding model.
    Loads data from specified path, preprocesses it, and uses lmfit to minimize residuals.
    Saves fitting report and plots results.
    Args:
        path (str): Path to the directory containing experimental data files.
        clip (int): Clipping value for X-axis data.
    """
    import base_utils.pipeline as pi
    import os

    files = pi.find_all_file(path)

    files = [f for f in files if f.endswith(".txt")]

    dataset = [pl.load_jasco_data(f)[0] for f in files]

    d: pl.XYData
    new_dataset = []
    for d in dataset:
        # d = d.xshift(273.15).normalize().yscale(-1).yshift(1)
        d = d.xshift(273.15).normalize()
        # conc = d.Title.split(" ")[1]
        # conc = conc.split("uM")[0]
        conc = d.Title.split("_")[2]
        conc = conc.split("microM")[0]

        new_dataset.append(d.rename_title(conc))
    dataset = new_dataset
    dataset_forfit = [da.clip(clip + 273.15, 100000) for da in dataset]
    # dataset_forfit = dataset.copy()
    fitparam = lf.Parameters()
    fitparam.add("deltaH", value=-91592, min=-158000, max=0)
    fitparam.add("deltaS", value=-200, min=-2000, max=-10)
    fitparam.add("deltaHnuc", value=-20238, min=-200000, max=-15000)
    fitparam.add("scaler0", value=1, min=0, max=2)
    fitparam.add("scaler1", value=1, min=0, max=2)
    fitparam.add("scaler2", value=1, min=0, max=2)
    fitparam.add("scaler3", value=1, min=0, max=2)
    # fitparam.add("scaler4", value=0.8, min=0, max=2)

    result = lf.minimize(
        fcn=objective, params=fitparam, args=(dataset_forfit,), max_nfev=1000000000
    )

    # save log
    R = 8.314
    Temp = 300
    deltaG = result.params["deltaH"] - Temp * result.params["deltaS"]
    print(deltaG)
    K_elong = np.exp(-deltaG / R / Temp)
    print(K_elong)
    K_c = K_elong * int(dataset[0].Title) / 1000000
    with open(os.path.join(path, str(clip), "fitting.log"), "w") as f:
        f.write(lf.fit_report(result))
        f.write("\n")
        f.write("parameters:::::::::::::::::::::::::::::::\n")
        f.write("\n")
        f.write("K_elong: " + str(K_elong) + "(300K)\n")
        f.write("K_c: " + str(K_c) + "(300K)\n")
        f.write("\n")

    # plot the result
    for i, d in enumerate(dataset):
        x = d.X
        y_forfit = model(
            result.params,
            dataset_forfit[i].X,
            c_tot=int(d.Title) / 1000000,
            scaler=result.params["scaler" + str(i)],
        )
        y_fit = model(
            result.params,
            dataset[i].X,
            c_tot=int(d.Title) / 1000000,
            scaler=result.params["scaler" + str(i)],
        )
        # print(x, y)
        scaler = result.params["scaler" + str(i)]
        y_forfit = y_forfit / scaler
        y_fit = y_fit / scaler
        d = d.yscale(1 / scaler)
        x = x - 273.15

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(4, 3)
        ax.plot(
            dataset[i].X - 273.15,
            y_fit,
            label="fit",
            color="blue",
            linestyle="--",
            # linewidth=0.7,
        )
        ax.plot(dataset_forfit[i].X - 273.15, y_forfit, label="fit", color="red")

        # circle (not filled)
        ax.scatter(x, d.Y, marker="o", facecolors="none", edgecolors="black")
        # ax.legend()
        ax.set_xlim(40, 100)
        # ax.set_ylim(-0.1, 1.1)
        ax.minorticks_on()
        file_name = os.path.join(path, str(clip), d.Title + ".svg")
        fig.savefig(file_name, bbox_inches="tight", transparent=True)


def fit2(path: str = "UV"):
    """
    Performs fitting of experimental data to the cooperative binding model with a background term.
    Loads data from specified path, preprocesses it, and uses lmfit to minimize residuals.
    Saves fitting report and plots results.
    Args:
        path (str): Path to the directory containing experimental data files.
    """
    import base_utils.pipeline as pi
    import os

    files = pi.find_all_file(path)
    clip = 40

    files = [f for f in files if f.endswith(".txt")]

    dataset = [pl.load_jasco_data(f)[0] for f in files]

    d: pl.XYData
    new_dataset = []
    for d in dataset:
        # d = d.xshift(273.15).normalize().yscale(-1).yshift(1)
        d = d.xshift(273.15).normalize()
        conc = d.Title.split(" ")[1]
        conc = conc.split("uM")[0]
        new_dataset.append(d.rename_title(conc))
    dataset = new_dataset
    dataset_forfit = [da.clip(clip + 273.15, 100000) for da in dataset]
    # dataset_forfit = dataset.copy()
    fitparam = lf.Parameters()
    fitparam.add("deltaH", value=-92708, min=-158000, max=0)
    fitparam.add("deltaS", value=-171, min=-2000, max=-10)
    fitparam.add("deltaHnuc", value=-32948, min=-40000, max=-15000)
    fitparam.add("scaler0", value=0.92160792, min=0, max=2)
    fitparam.add("scaler1", value=1, min=0, max=2)
    fitparam.add("scaler2", value=0.88709964, min=0, max=2)
    fitparam.add("scaler3", value=0.91145068, min=0, max=2)
    # fitparam.add("scaler4", value=0.8, min=0, max=2)

    fitparam.add("background0", value=-0.1, min=-0.1, max=1)
    fitparam.add("background1", value=-0.1, min=-0.1, max=1)
    fitparam.add("background2", value=-0.1, min=-0.1, max=1)
    fitparam.add("background3", value=-0.1, min=-0.1, max=1)
    # fitparam.add("background4", value=0, min=-0.1, max=0.1)

    result = lf.minimize(
        fcn=objective2, params=fitparam, args=(dataset_forfit,), max_nfev=1000000000
    )

    # save log
    R = 8.314
    Temp = 300
    deltaG = result.params["deltaH"] - Temp * result.params["deltaS"]
    print(deltaG)
    K_elong = np.exp(-deltaG / R / Temp)
    print(K_elong)
    K_c = K_elong * int(dataset[0].Title) / 1000000
    with open(os.path.join(path, str(clip), "fitting.log"), "w") as f:
        f.write(lf.fit_report(result))
        f.write("\n")
        f.write("parameters:::::::::::::::::::::::::::::::\n")
        f.write("\n")
        f.write("K_elong: " + str(K_elong) + "(300K)\n")
        f.write("K_c: " + str(K_c) + "(300K)\n")
        f.write("\n")

    # plot the result
    for i, d in enumerate(dataset):
        x = d.X
        y_forfit = model(
            result.params,
            dataset_forfit[i].X,
            c_tot=int(d.Title) / 1000000,
            scaler=result.params["scaler" + str(i)],
        )
        y_fit = model(
            result.params,
            dataset[i].X,
            c_tot=int(d.Title) / 1000000,
            scaler=result.params["scaler" + str(i)],
        )
        # print(x, y)
        scaler = result.params["scaler" + str(i)]
        y_forfit = y_forfit / scaler
        y_fit = y_fit / scaler
        d = d.yscale(1 / scaler).yshift(-result.params["background" + str(i)])
        x = x - 273.15

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(4, 3)
        # ax.plot(
        #     dataset[i].X - 273.15,
        #     y_fit,
        #     label="fit",
        #     color="blue",
        #     linestyle="--",
        #     # linewidth=0.7,
        # )
        ax.plot(dataset_forfit[i].X - 273.15, y_forfit, label="fit", color="red")

        # circle (not filled)
        ax.scatter(x, d.Y, marker="o", facecolors="none", edgecolors="black")
        # ax.legend()
        ax.set_xlim(40, 100)
        ax.set_ylim(-0.1, 1.1)
        ax.minorticks_on()
        file_name = os.path.join(path, str(clip), d.Title + ".svg")
        fig.savefig(file_name, bbox_inches="tight", transparent=True)


def test_plot():
    """
    Generates a test plot for the cooperative binding model across different concentrations.
    """
    conc = [
        20,
        40,
        60,
        80,
        100,
    ]  # in uM
    temp = np.linspace(0, 100, 1000)
    deltaH = -121592
    deltaS = -252

    deltaHnuc = -19787
    scaler = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for c in conc:
        y = temp_cooperative(
            temp + 273.15, deltaH, deltaS, deltaHnuc, c / 1000000, scaler=scaler
        )
        ax.plot(temp, y, label=f"{c} uM")
    # ax.set_xlabel("Temperature (°C)")
    # ax.set_ylabel("Cooperati")
    ax.set_xlim(60, 100)
    # ax.set_title("Cooperative Binding vs Temperature")
    # ax.legend()

    # plt.show()
    fig.savefig("cooperative_binding.svg", bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    # main()
    # fit("26_2",`r` clip=65)

    test_plot()
