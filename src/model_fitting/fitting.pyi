from matplotlib import pyplot as plt
import numpy as np
import lmfit as lf
import base_utils.plotlib as pl
import numpy.typing as npt

def isodesmic(X, K): ...
def cubic(b, c, d) -> float: ...
def cooperative(Conc, K, sigma, scaler=1) -> np.ndarray:
    """
    Conc: total concentration of substrate
    K: equilibrium constant
    sigma: cooperativity factor

    return: rate of supramolecular polymer formation
    """

def cooperative_stabilized(
    Conc: float | npt.NDArray[np.number],
    K: float | np.number,
    sigma: float | np.number,
    scaler: float | np.number = 1,
) -> float | npt.NDArray[np.number]:
    """
    Conc: total concentration of substrate
    K: equilibrium constant
    sigma: cooperativity factor

    return: rate of supramolecular polymer formation
    """

def temp_cooperative(
    Temp, deltaH, deltaS, deltaHnuc, c_tot, scaler=1
) -> np.ndarray: ...
def model(
    params: lf.Parameters, x: np.ndarray, c_tot: float, scaler: float
) -> np.ndarray: ...
def objective(params: lf.Parameters, data: list[pl.XYData]) -> np.ndarray:
    """
    params: parameters for the model
    data: mesured x-y dataset

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
    params: parameters for the model
    data: mesured x-y dataset

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
