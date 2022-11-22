import pandas as pd
from numpy import sin, cos, arcsin, arctan
import numpy as np


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii
        for the input damage levels

    Examples
    --------

    >>> import armageddon
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
                   'outcome': 'Airburst'}
    >>> armageddon.damage_zones(outcome, 52.79, -2.95, 135,
                                pressures=[1e3, 3.5e3, 27e3, 43e3])
    """

    r_h = outcome['burst_distance']
    Ek = outcome['burst_energy']
    zb = outcome['burst_altitude']
    Rp = 6371e3
    pressures = np.array(pressures)

    sin_blat = ((sin(lat) * cos(r_h / Rp)) +
                (cos(lat) * sin(r_h / Rp) * cos(bearing)))
    blat = arcsin(sin_blat)

    tan_blon_diff = ((sin(bearing) * sin(r_h / Rp) * cos(lat)) /
                     (cos(r_h / Rp) - (sin(lat) * sin(blat))))
    blon = arctan(tan_blon_diff) + lon

    discriminant = np.sqrt((3.24e14 + (1.256e12 * pressures)))
    pre_sol = (((((-1.8e7 + discriminant) / 6.28e11)**(-2/1.3)) *
                (Ek**(2/3))) - (zb**2))
    damrad = np.sqrt(pre_sol)

    return blat, blon, damrad


fiducial_means = {'radius': 35, 'angle': 45, 'strength': 1e7,
                  'density': 3000, 'velocity': 19e3,
                  'lat': 53.0, 'lon': -2.5, 'bearing': 115.}
fiducial_stdevs = {'radius': 1, 'angle': 1, 'strength': 5e6,
                   'density': 500, 'velocity': 1e3,
                   'lat': 0.025, 'lon': 0.025, 'bearing': 0.5}


def impact_risk(planet, means=fiducial_means, stdevs=fiducial_stdevs,
                pressure=27.e3, nsamples=100, sector=True):
    """
    Perform an uncertainty analysis to calculate the risk for each affected
    UK postcode or postcode sector

    Parameters
    ----------
    planet: armageddon.Planet instance
        The Planet instance from which to solve the atmospheric entry

    means: dict
        A dictionary of mean input values for the uncertainty analysis. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    stdevs: dict
        A dictionary of standard deviations for each input value. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    pressure: float
        A single pressure at which to calculate the damage zone for each impact

    nsamples: int
        The number of iterations to perform in the uncertainty analysis

    sector: logical, optional
        If True (default) calculate the risk for postcode sectors, otherwise
        calculate the risk for postcodes

    Returns
    -------
    risk: DataFrame
        A pandas DataFrame with columns for postcode (or postcode sector) and
        the associated risk. These should be called ``postcode`` or ``sector``,
        and ``risk``.
    """

    if sector:
        return pd.DataFrame({'sector': '', 'risk': 0}, index=range(1))
    else:
        return pd.DataFrame({'postcode': '', 'risk': 0}, index=range(1))
