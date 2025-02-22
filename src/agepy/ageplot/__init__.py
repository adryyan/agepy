"""Plotting module containing custom AGE matplotlib styles and functions
for creating nice plots. 

Attributes
----------
age_styles: list
    List of strings containing all the available custom AGE style 
    sheets.
mpl_styles: list
    List of strings containing all the available matplotlib style 
    sheets.
colors: list
    List of AGE colors (taken from the seaborn colorblind palette). 
"""

import matplotlib.pyplot as plt

age_styles = ["age", "pccp", "powerpoint", "latexbeamer"]
mpl_styles = ["default"]
mpl_styles.extend(plt.style.available)

colors = [
    "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161",
    "#fbafe4", "#949494", "#ece133", "#56b4e9"
]


def use(styles):
    """Function calling :py:func:`plt.style.use` for easier access to 
    the custom AGE matplotlib style sheets.

    Notes
    -----
    All style rcParams are reset to the matplotlib default before
    loading the specified styles.

    Warnings
    --------
    Compatibility between styles is not guaranteed.

    Parameters
    ----------
    styles: str or list of string
        Styles to be loaded using :py:func:`plt.style.use`. Available styles
        can be viewed by calling :py:data:`ageplot.age_styles` and
        :py:data:`ageplot.mpl_styles`.

    Examples
    --------
    How to use the AGE style:

    >>> from agepy import ageplot
    >>> ageplot.use("age")

    """
    load_styles = []

    # Check if styles are available
    if isinstance(styles, list):
        for style in styles:
            if style in age_styles:
                load_styles.append("agepy.ageplot." + style)
            elif style in mpl_styles:
                load_styles.append(style)
            else:
                raise ValueError(style + " is not an available style.")
    elif isinstance(styles, str):
        if styles in age_styles:
            load_styles.append("agepy.ageplot." + styles)
        elif styles in mpl_styles:
            load_styles.append(styles)
        else:
            raise ValueError(styles + " is not an available style.")
    else:
        raise TypeError("Expected str or list of strings specifying styles.")

    plt.style.use("default")  # reset rcParams before applying the style
    plt.style.use(load_styles)  # apply the selected styles


class figsize():
    """Class for choosing the appropriate size of matplotlib figures.
    
    This class provides access to the width and height of the available
    space in different media in order to choose a figure size for
    matplotlib plots.

    Parameters
    ----------
    medium: str, optional
        Name of the medium. Media, for which the size is
        implemented, can be viewed with :py:attr:`figsize.media`. 
        Default: None

    Attributes
    ----------
    w: float
        Recommended width for a figure. Most of the time this will be
        equivalent to the full available width.
    h: float
        Recommended height corresponding to the width 
        (width * 3 / 4).
    wh: tuple
        Tuple (w, h) containing the width and recommended height.
        If there is a corresponding style sheet, the default figure
        size will be set to (w, h) by :py:func:`ageplot.use`.
    hmax: float
        Height in inches available for a figure.

    Examples
    --------
    A simple example for the PCCP journal:

    >>> from agepy import ageplot
    >>> my_figsize = ageplot.figsize("pccp")
    >>> my_figsize.wh
    (3.54, 2.6550000000000002)

    This could be used for a matplotlib plot like this:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=my_figsize.hw)

    """

    # (textwidth, textheight)
    # If the journal uses multiple columns, columnwidth instead of
    # textwidth is used.
    _pagesizes = {
        "pccp": (3.54, 9.54),
        "powerpoint": (5, 5.625),  # 0.5 of full width
        "latexbeamer": (2.766, 3.264)  # 0.5 of full width
    }
    media = list(_pagesizes.keys())

    def __init__(self, medium=None):
        if medium is None:
            self.w = 6.4
            self.h = 4.8
            self.hmax = None
        else:
            self.w = self._pagesizes[medium][0]
            self.h = self._pagesizes[medium][0] * 0.75
            self.hmax = self._pagesizes[medium][1]

        self.wh = (self.w, self.h)

    @property
    def media(self):
        """Class wide attribute containing a list of media, for which 
        the figsize is implemented.

        """
        return self.media
