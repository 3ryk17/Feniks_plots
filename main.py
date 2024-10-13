from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

'''
Script for plotting
You need following packages:
        numpy
        pandas
        matplotlib
        scipy
        scienceplots
'''

'''
_________________TO_DO_________________
finish write function
fix cubic interpolation
add error margin to experimental data function

'''

class Polynomial:
    # defines polynomials
    def __init__(self, *args: float):
        self.polynomial_factors = []
        for k in args:
            self.polynomial_factors.append(k)
        self.length = len(self.polynomial_factors)

    def value(self, x) -> float:
        val = 0
        v = 0
        for i in range(self.length):
            val = v+self.polynomial_factors[i]*(x**i)
            v = val
        return val


def read_data(file_path, x_name: str) -> tuple:
    # Read data from a csv file
    d = pd.read_csv(file_path)
    x = d[x_name].to_numpy()

    return x

def legend_set(ax, location='upper right', number_of_columns=1) -> None:
    # legend settings
    ax.legend(loc=location, fontsize=14, ncol=number_of_columns)


def basic_style() -> None:
    # define basic plot style
    plt.style.use(['science', 'grid', 'no-latex'])
    plt.rc('font', family='Times new roman')

def basic_style2(axes, row: int, column: int, x_label: str, y_label: str, title: str, x_limit: list = None, y_limit: list = None, ncol=1, lloc='upper right') -> None:
    # basic settings for plotting
    axis = axes[row, column]
    axis.set_xlabel(x_label, fontsize='16')
    axis.set_ylabel(y_label, fontsize='16')
    axis.set_title(title, fontsize='20')
    axis.set_xlim(x_limit)
    axis.set_ylim(y_limit)
    legend_set(axis, number_of_columns=ncol, location=lloc)

def muted_style() -> None:
    # define muted style
    plt.style.use(['science', 'grid', 'no-latex', 'muted'])
    plt.rc('font', family='Arial')

def write(text: str) -> None:
    pass


def experimental_data(axes, row: int, column: int, x, y, c='black', l='pomiary') -> None:
    # default settings for plotting experimental data
    axis = axes[row, column]
    axis.plot(x, y, 'o', color=c, lw='0.7', ms=9, label=l)


def log_scale(axes, row: int, column: int, x, y, c='black', l='pomiary') -> None:
    # log scale plotting
    axis = axes[row, column]
    # set log scale on x axis
    axis.set_xscale('log')
    # set log scale on y axis
    axis.set_yscale('log')
    axis.plot(x, y, 'o', color=c, lw='1.5', ms=9, label=l)

def cubic_interpolation(axes, row: int, column: int, x, y) -> None:  # doesn't work yet!!!
    axis = axes[row, column]
    y_f = interp1d(x, y, 'cubic')
    x_interp = np.linspace(1, 1, 10)
    y_interp = y_f(x_interp)
    axis.plot(x_interp, y_interp, '-', color='pink', lw='0.8', ms=6, label='cubic interpolation')

def function_for_fitting(x, vn) -> float:
    y_value = vn/(4*x)
    return y_value

def improved_function_for_fitting(x, vn) -> float:
    radius_of_the_test_tube = 0.008
    y_value = vn/(4*(x+0.6*radius_of_the_test_tube))
    return y_value

def curve_fitting(axes, row: int, column: int, x, y, starting_parameters=None, f=function_for_fitting, c='pink', l='teoria') -> None:
    """
    Change function_for_fitting as you like:
    function_for_fitting(x, a, b, c ...)

    x - variable
    a, b, c ... - constants to fit
    or add your own function for parameter f
    """
    axis = axes[row, column]
    p_optimal, p_covariance = curve_fit(f, x, y, p0=starting_parameters)
    x_model = np.linspace(min(x), max(x), 100)
    y_model = f(x_model, *p_optimal)
    axis.plot(x_model, y_model, '-', color=c, lw='1.0', ms=6, label=l)
    print(f'optimal parameters: {p_optimal}, covariance: {p_covariance}')

def frequency_length_mod_3d() -> None:
    pass

def log_scale_curve_fitting(axes, row: int, column: int, x, y, starting_parameters=None, f=function_for_fitting, c='pink', l='teoria') -> None:
    axis = axes[row, column]
    # set log scale on x axis
    axis.set_xscale('log')
    # set log scale on y axis
    axis.set_yscale('log')
    axis = axes[row, column]
    p_optimal, p_covariance = curve_fit(f, x, y, p0=starting_parameters)
    x_model = np.linspace(min(x), max(x), 100)
    y_model = f(x_model, *p_optimal)
    axis.plot(x_model, y_model, '-', color=c, lw='1.0', ms=6, label=l)


def main() -> None:
    data_file_path = "data.csv"
    x_data = read_data(data_file_path, 'L')
    y_data_mod1 = read_data(data_file_path, 'mod 1')
    y_data_mod2 = read_data(data_file_path, 'mod 2')
    y_data_mod3 = read_data(data_file_path, 'mod 3')
    y_data_mod4 = read_data(data_file_path, 'mod 4')

    number_of_columns = 2
    number_of_rows = 2
    width = 32
    height = 16

    title = 'Częstotliwość od długości z ulepszoną teorią'
    x_label = 'długość (L) [m]'
    y_label = 'częstotliwość (f) [Hz]'

    title_2 = 'Skala logarytmiczna'
    x_label_2 = 'długość'
    y_label_2 = 'częstotliwość'

    title_3 = 'Zestawienie pomiarów dla różnych modów'
    x_label_3 = 'długość (L) [m]'
    y_label_3 = 'częstotliwość (f) [Hz]'

    title_4 = 'Częstotliwość od długości'
    x_label_4 = 'długość (L) [m]'
    y_label_4 = 'częstotliwość (f) [Hz]'

    basic_style()
    fig, axes = plt.subplots(number_of_rows, number_of_columns, figsize=(width, height), squeeze=False)

    experimental_data(axes, row=0, column=1, x=x_data, y=y_data_mod1, l='pomiar mod 1')
    curve_fitting(axes, row=0, column=1, x=x_data, y=y_data_mod1, starting_parameters=346, c='blue')
    curve_fitting(axes, row=0, column=1, x=x_data, y=y_data_mod1, starting_parameters=346, f=improved_function_for_fitting, c='red', l='ulepszona teoria')

    log_scale(axes, row=1, column=1, x=x_data, y=y_data_mod1, l='pomiar mod 1')
    log_scale_curve_fitting(axes, row=1, column=1, x=x_data, y=y_data_mod1, starting_parameters=346, c='blue')
    log_scale_curve_fitting(axes, row=1, column=1, x=x_data, y=y_data_mod1, starting_parameters=346, f=improved_function_for_fitting, c='red', l='ulepszona teoria')

    experimental_data(axes, row=1, column=0, x=x_data, y=y_data_mod1, c='purple', l='pomiary mod 1')
    experimental_data(axes, row=1, column=0, x=x_data, y=y_data_mod2, c='red', l='pomiary mod 2')
    experimental_data(axes, row=1, column=0, x=x_data, y=y_data_mod3, c='orange', l='pomiary mod 3')
    experimental_data(axes, row=1, column=0, x=x_data, y=y_data_mod4, c='yellow', l='pomiary mod 4')
    curve_fitting(axes, row=1, column=0, x=x_data, y=y_data_mod1, c='#4b0d70', l='teoria mod 1')
    curve_fitting(axes, row=1, column=0, x=x_data, y=y_data_mod2, c='#780606', l='teoria mod 2')
    curve_fitting(axes, row=1, column=0, x=x_data, y=y_data_mod3, c='#94611f', l='teoria mod 3')
    curve_fitting(axes, row=1, column=0, x=x_data, y=y_data_mod4, c='#8a800b', l='teoria mod 4')

    experimental_data(axes, row=0, column=0, x=x_data, y=y_data_mod1, l='pomiar mod 1')
    curve_fitting(axes, row=0, column=0, x=x_data, y=y_data_mod1, starting_parameters=346, c='blue')

    """

    log_scale(axes, row=1, column=1, x=x_data, y=y_data_mod1, c='purple', l='pomiary mod 1')
    log_scale(axes, row=1, column=1, x=x_data, y=y_data_mod2, c='red', l='pomiary mod 2')
    log_scale(axes, row=1, column=1, x=x_data, y=y_data_mod3, c='orange', l='pomiary mod 3')
    log_scale(axes, row=1, column=1, x=x_data, y=y_data_mod4, c='yellow', l='pomiary mod 4')
    log_scale_curve_fitting(axes, row=1, column=1, x=x_data, y=y_data_mod1, c='#4b0d70', l='teoria mod 1')
    log_scale_curve_fitting(axes, row=1, column=1, x=x_data, y=y_data_mod2, c='#780606', l='teoria mod 2')
    log_scale_curve_fitting(axes, row=1, column=1, x=x_data, y=y_data_mod3, c='#94611f', l='teoria mod 3')
    log_scale_curve_fitting(axes, row=1, column=1, x=x_data, y=y_data_mod4, c='#8a800b', l='teoria mod 4')
    
    """

    basic_style2(axes, row=0, column=1, x_label=x_label, y_label=y_label, title=title)
    basic_style2(axes, row=1, column=1, x_label=x_label_2, y_label=y_label_2, title=title_2)
    basic_style2(axes, row=1, column=0, x_label=x_label_3, y_label=y_label_3, title=title_3, ncol=2)
    basic_style2(axes, row=0, column=0, x_label=x_label_4, y_label=y_label_4, title=title_4)

    plt.show()


if __name__ == '__main__':
    main()

