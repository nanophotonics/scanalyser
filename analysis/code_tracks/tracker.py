"""
@Created : 19/11/2020
@Edited  : 31/05/2022
@Author  : Alex Poppe
@File    : tracker.py
@Software: Pycharm
@Description:
Classes that store any necessary information for each Track or Event
"""
import numpy as np
from analysis.code_tracks.detector import bounds_generator


# Individual Track
class Track(object):
    """ Class that stores each sequential set of detected outliers (peaks) as an object
    """

    def __init__(self):
        """ Initialise track lists and variables
        """
        self.group_id = -1  # instantiate PH group ID
        self.trace = []  # instantiate list of track locations for current object (i.e. history of where it has been)
        self.trace_bounds = []  # instantiate list of bound locations (left & right edges of each peak (in pixels))
        self.trace_time = []  # instantiate list of time steps for current object (i.e. history of when it existed)
        self.zip_count = 0  # instantiate the count of the number of times this track has been zipped with others


# All Tracks in one scan
class Tracker(object):
    """ Class that updates each track object through peak assignments, zips, or others manipulations, for a single scan
    """

    def __init__(self, particle_num):
        """ Initialise variables

        Args:
            particle_num: Str, the particle number that the current tracker is storing information for
        """
        self.particle_num = particle_num
        self.tracks = []  # instantiate a list of all tracks in the current scan

    def update(self, tracks):
        """ Function that takes in input 2D array of tracks and converts it into individual track objects which are then
        output and used inside an existing tracker object

        Args:
            tracks: Array, The 2D array containing individual rudimentary tracks, differentiated by numerical labels
        """
        # Collect the unique track IDs (removing 0, as it is the background (i.e. not a track))
        uniques = np.delete(np.unique(tracks), 0)

        # Cycle through each track ID
        for unique in uniques:
            # Form a new track for the current track ID
            track = Track()

            # Extract the [time step, wavenumber] coordinate pairs from the array
            coords = np.argwhere(tracks == unique)

            # Generate the centroids, bounds, and time steps for the current coordinate set
            track_centroids, track_bounds, track_times = bounds_generator(coords=coords)

            # Assign the track information to the current track
            track.trace.extend(track_centroids)
            track.trace_bounds.extend(track_bounds)
            track.trace_time.extend(track_times)

            # Append the track to the list of tracks within the tracker object
            self.tracks.append(track)
        return


# All Groups in one scan
class Groups(object):
    """ Class that stores all Group feature vectors from one scan
    """

    def __init__(self, particle_num=-1):
        """ Initialise variables
        Args:
            particle_num: Str, The particle number (e.g. '0506', '1045')
        """
        self.particle_num = particle_num  # ID of the event
        self.vectors = []  # list of all Group feature vectors*
        # *These contain information about a Group such as: duration, start/end-points, particle number, number of
        # Tracks, difference spectra, ...


# Single Event
class Event(object):
    """ Class that stores all Group feature vectors from one Event
    """

    def __init__(self, event_id=-1):
        """ Initialise variables
        Args:
            event_id: int, the ID of the event
        """
        self.id = event_id  # ID of the event
        self.vectors = []  # list of all Group feature vectors*
        # *These contain information about a Group such as: duration, start/end-points, particle number, number of
        # Tracks, difference spectra, ...

        self.particles = []  # list of all particles
        self.duration_stats = []  # List containing the [min, max] detected durations
        self.intra_strength = None  # The 'mean +- std' strength (i.e. the Wasserstein distance)
        self.mean_difference = None  # The mean difference spectrum
        self.mean_spectrum = None  # The mean raw spectrum


# All Events
class Events_Tracker(object):
    """ Class that stores all events from the ID_Tracker grouped by the Event clustering task
    """

    def __init__(self):
        self.events = []  # List of all Events
        self.num_events = 0  # Number of Events
        self.top_events = []  # List of most common events in descending order of occurrence

        self.affinity_matrix = None  # Affinity matrix of the Groups
        self.distance_matrix = None  # Distance matrix of the Groups
        self.labels = None  # Cluster labels of the multi-peak Groups
        self.silhouette = None  # Silhouette sample scores for each Event*
        # *mean +- std Silhouette sample scores are found in '/events/[n_clusters]/event_summary.txt'
