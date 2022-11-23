import os

import numpy as np
import pandas as pd


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='tabular',
                 atmos_filename=os.sep.join((os.path.dirname(__file__), '..',
                                             'resources',
                                             'AltitudeDensityTable.csv')),
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3,
                 Rp=6371e3, g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'

        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename

        try:
            # set function to define atmoshperic density
            if atmos_func == 'exponential':
                self.rhoa = lambda x: rho0 * np.exp(-x/H)
            elif atmos_func == 'tabular':
                self.rhoa = self.create_tabular_density(atmos_filename)
            elif atmos_func == 'constant':
                self.rhoa = lambda x: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be 'exponential', 'tabular' or 'constant'"
                )
        except NotImplementedError:
            print("atmos_func {} not implemented yet.".format(atmos_func))
            print("Falling back to constant density atmosphere for now")
            self.rhoa = lambda x: rho0

    def f_solver1(self, t, y, density):

        f = np.zeros_like([1, 1, 1, 1, 1, 1], dtype=float)
        # y = np.array([velocity,(4*np.pi*density*radius**3)/3, angle, init_altitude, 0, radius])
        f[0] = self.g*np.sin(y[2]) + (-self.Cd * self.rhoa(y[3]) *
                                      (np.pi * y[5]**2) * y[0]**2) / (2 * y[1])
        f[1] = (-self.Ch * self.rhoa(y[3]) *
                (np.pi * y[5]**2) * y[0]**3) / (2 * self.Q)
        f[2] = (self.g * np.cos(y[2]) / y[0]) - (((self.Cl * self.rhoa(y[3])
                                                   * (np.pi * y[5]**2) * y[0])) / (2 * y[1])) - ((y[0] * np.cos(y[2]))/(self.Rp + y[3]))
        f[3] = - y[0] * np.sin(y[2])
        f[4] = y[0] * np.cos(y[2]) / (1 + (y[3] / self.Rp))
        f[5] = y[0] * np.sqrt(7 * self.alpha * self.rhoa(y[3])/(2 * density))

        return f

    def f_solver(self, t, y, density):

        f = np.zeros_like([1, 1, 1, 1, 1, 1], dtype=float)
        # y = np.array([velocity,(4*np.pi*density*radius**3)/3, angle, init_altitude, 0, radius])
        f[0] = self.g*np.sin(y[2]) + (-self.Cd * self.rhoa(y[3]) *
                                      (np.pi * y[5]**2) * y[0]**2) / (2 * y[1])
        f[1] = (-self.Ch * self.rhoa(y[3]) *
                (np.pi * y[5]**2) * y[0]**3) / (2 * self.Q)
        f[2] = (self.g * np.cos(y[2]) / y[0]) - (((self.Cl * self.rhoa(y[3])
                                                   * (np.pi * y[5]**2) * y[0])) / (2 * y[1])) - ((y[0] * np.cos(y[2]))/(self.Rp + y[3]))
        f[3] = - y[0] * np.sin(y[2])
        f[4] = y[0] * np.cos(y[2]) / (1 + (y[3] / self.Rp))
        f[5] = 0

        return f

    def create_tabular_density(self, atmos_filename):
        """
        Create a function given altitude return the density of atomosphere
        using tabulated value
        Parameters
        ----------
        filename : str, optional
            Path to the tabular. default="./resources/AltitudeDensityTable.csv"
        Returns
        -------
        tabular_density : function
            A function that takes altitude as input and return the density of
            atomosphere density at given altitude.
        """
        X = []
        Y = []
        data = pd.read_csv(atmos_filename)
        for i in data[data.keys()[0]]:
            temp = i.split()
            X.append(eval(temp[0]))
            Y.append(eval(temp[1]))

        def tabular_density(x):
            if x > X[-1]:
                return 0
            for i in range(len(X)):
                if X[i] >= x:
                    break
            pressure = (x - X[i-1])/(X[i] - X[i-1]) * (Y[i] - Y[i-1]) + Y[i-1]

            return pressure
        return tabular_density

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        Returns
        -------
        Result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """
        if radians == False:
            angle = np.radians(angle)
            y = np.array([velocity, (4*np.pi*radius**3)*density /
                          3, angle, init_altitude, 0, radius])
        else:
            y = np.array([velocity, (4*np.pi*radius**3)*density /
                          3, angle, init_altitude, 0, radius])

        t = np.array(0)
        y_all = [y]
        t_all = [t]

        while y[3] > 0:

            if (1.2 * (np.exp(-y[3]/8000)) * y[0]**2) >= strength:

                k1 = dt * self.f_solver1(t, y, density)
                k2 = dt * self.f_solver1(t + 0.5*dt, y + 0.5*k1, density)
                k3 = dt * self.f_solver1(t + 0.5*dt, y + 0.5*k2, density)
                k4 = dt * self.f_solver1(t + dt, y + k3, density)
                y = y + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
                y_all.append(y)
                t = t + dt
                t_all.append(t)

            else:
                k1 = dt * self.f_solver(t, y, density)
                k2 = dt * self.f_solver(t + 0.5*dt, y + 0.5*k1, density)
                k3 = dt * self.f_solver(t + 0.5*dt, y + 0.5*k2, density)
                k4 = dt * self.f_solver(t + dt, y + k3, density)
                y = y + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
                y_all.append(y)
                t = t + dt
                t_all.append(t)

        yy = pd.DataFrame(
            y_all, columns=['velocity', 'mass', 'angle', 'altitude', 'distance', 'radius'])

        if radians == False:
            yy.angle = yy.angle * 180 / np.pi
            yy['time'] = t_all
            result = yy.drop(index=len(yy.index)-1)

        else:
            yy['time'] = t_all
            result = yy.drop(index=len(yy.index)-1)

        return result

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude

        """

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        
        result1 = result.copy()
        mass = result1["mass"]
        velocity = result1["velocity"]
        altitude = np.array(result1["altitude"])
        dezd = np.array(0.5 * mass * velocity**2)
        temp = 1000*(dezd[1:] - dezd[:-1])/(altitude[:-1]- altitude[1:])
        temp = np.insert(temp, 0, 0)
        result1.insert(len(result1.columns),
                      'dedz', -temp)
        return result1


        return result1

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats

        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time

        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the key:
                ``outcome`` (which should contain one of the
                following strings: ``Airburst`` or ``Cratering``),
            as well as the following 4 keys:
                ``burst_peak_dedz``, ``burst_altitude``,
                ``burst_distance``, ``burst_energy``
        """
        result = self.calculate_energy(result)
        result1 = result.dedz
        burstidx = result1.idxmax()
        initial_energy = 0.5 * result["mass"][0] * result["velocity"][0]**2
        burstenergy = 0.5 * result["mass"][burstidx] * result["velocity"][burstidx]**2
        outcome = "Airburst"
        if burstidx == len(result) - 1:
            outcome = "Cratering"
            burstenergy = max(burstenergy, initial_energy - burstenergy)
        else:
            burstenergy = initial_energy - burstenergy
        outcome = {'outcome': outcome,
                   'burst_peak_dedz': result['dedz'][burstidx],
                   'burst_altitude': result["altitude"][burstidx],
                   'burst_distance': result["distance"][burstidx],
                   'burst_energy': burstenergy}
        return outcome

        
