"""Module dealing with postcode information."""

import numpy as np

__all__ = ['PostcodeLocator', 'great_circle_distance']


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)}
    >>> with numpy.printoptions(formatter={'all', fmt}):
        print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """
    Rp = 6371000
    
    
    latlon1 = np.array(latlon1)
    latlon2 = np.array(latlon2)
    
    if latlon1.ndim == 1:
        latlon1 = latlon1.reshape(1,2)

    if latlon2.ndim == 1:
        latlon2 = latlon2.reshape(1,2)
    
    distance = np.empty((len(latlon1), len(latlon2)), float)
    lat1 = latlon1[:, 0]
    lat2 = latlon2[:, 0]
    lon1 = latlon1[:, 1]
    lon2 = latlon2[:, 1]
    #print(lat1)

    for i in range(len(latlon1)):
        
        for j in range(len(latlon2)):
            #print(lat1[i])
            # print(lat2[j])
            # print(lon1[i])
            # print(lon2[j])
            #dis = 2*Rp*np.arcsin(np.sqrt((np.sin(abs(lat1[i]-lat2[j])/2))**2+np.cos(lat1[i])*np.cos(lat2[j])*(np.sin(abs(lon1[i]-lon2[j])/2))**2))
            #dis = Rp*np.arccos(np.sin(lat1[i])*np.sin(lat2[j])+np.cos(lat1[i])*np.cos(lat2[j])*np.cos(abs(lon1[i]-lon2[j])))
            num = np.sqrt((np.cos(lat2[j])*np.sin(abs(lon1[i]-lon2[j])))**2+(np.cos(lat1[i])*np.sin(lat2[j])-np.sin(lat1[i])*np.cos(lat2[j])*np.cos(abs(lon1[i]-lon2[j])))**2)
            den = np.sin(lat1[i])*np.sin(lat2[j])+np.cos(lat1[i])*np.cos(lat2[j])*np.cos(abs(lon1[i]-lon2[j]))
            dis = Rp*np.arctan(num/den)
            print(num, den)
            
            distance[i][j] = dis
    return distance

#fmt = lambda x: np.format_float_scientific(x, precision=3)
#with np.printoptions(formatter={'all', fmt}):
    #print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])
print(great_circle_distance(pnts1, pnts2))
#print(np.array([55, 1.0]).reshape(1,2))


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='',
                 census_file='',
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .csv file containing census data by postcode sector.

        norm : function
            Python function defining the distance between points in
            latitude-longitude space.

        """
        self.norm = norm

    def get_postcodes_by_radius(self, X, radii, sector=False):
        """
        Return (unit or sector) postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X
        sector : bool, optional
            if true return postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements
            of radii to the location X.


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3])
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773),
                                            [0.4e3, 0.2e3], True)
        """

        return [[]]

    def get_population_of_postcode(self, postcodes, sector=False):
        """
        Return populations of a list of postcode units or sectors.

        Parameters
        ----------
        postcodes : list of lists
            list of postcode units or postcode sectors
        sector : bool, optional
            if true return populations for postcode sectors,
            otherwise returns populations for postcode units

        Returns
        -------
        list of lists
            Contains the populations of input postcode units or sectors


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_population_of_postcode([['SW7 2AZ', 'SW7 2BT',
                                                 'SW7 2BU', 'SW7 2DD']])
        >>> locator.get_population_of_postcode([['SW7  2']], True)
        """

        return [[]]
