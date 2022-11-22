import folium
import os


def plot_circle(lat, lon, radius, map=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> armageddon.plot_circle(52.79, -2.95, 1e3, map=None)
    """
    #os.remove("map.html")
    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)
        
    folium.Circle([lat, lon], radius, fill=True,
                   fillOpacity=0.8, **kwargs).add_to(map)
    map.save("map.html")
    
    return map

plot_circle(52.79, -2.95, 1e3, map=None)
