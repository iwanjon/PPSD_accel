from obspy.signal import PPSD # Import the original class
import numpy as np
from pathlib import Path
from obspy.imaging.cm import obspy_sequential
from obspy.core.util import AttribDict
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FormatStrFormatter
import warnings
from matplotlib.colors import LinearSegmentedColormap



from obspy.signal.spectral_estimation import (
    get_idc_infra_hi_noise,
    get_idc_infra_low_noise,
    get_nhnm,
    get_nlnm,
    earthquake_models
)

ACC_NOISE_MODEL_FILE = Path(__file__).parent / "custom_accel_noise_model.npz"

class CustomPPSD(PPSD):

    def __init__(self, stats, metadata, *args, special_handling=None, **kwargs):
            """
            Custom PPSD that allows 'accelerometer' as a special_handling type.
            """
            
            # 1. Check if the user wants 'accelerometer'
            is_accelerometer = False
            if special_handling and special_handling.lower() == 'accelerometer':
                is_accelerometer = True
                
                # Temporarily set it to None so the parent class PPSD 
                # doesn't throw a ValueError during initialization!
                special_handling = None 
                
            # 2. Run the original PPSD setup normally
            # We pass *args and **kwargs to ensure all other settings (like ppsd_length, 
            # db_bins, etc.) are passed through perfectly without having to re-type them.
            super().__init__(stats, metadata, *args, special_handling=special_handling, **kwargs)
            
            # 3. Overwrite the attribute internally for your custom class
            if is_accelerometer:
                self.special_handling = 'accelerometer'

    def get_alnm(self):
        data = np.load(ACC_NOISE_MODEL_FILE)
        periods = data['model_periods']
        nlnm = data['low_noise']
        return (periods, nlnm)

    def get_ahnm(self):
        data = np.load(ACC_NOISE_MODEL_FILE)
        periods = data['model_periods']
        nlnm = data['high_noise']
        return (periods, nlnm)

    def plot(self, filename=None, show_coverage=True, show_histogram=True,
             show_percentiles=False, percentiles=[0, 25, 50, 75, 100],
             show_noise_models=True, grid=True, show=True,
             max_percentage=None, period_lim=(0.01, 179), show_mode=False,
             show_mean=False, cmap=obspy_sequential, cumulative=False,
             cumulative_number_of_colors=20, xaxis_frequency=False,
             show_earthquakes=None):
        """
        Plot the 2D histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        .. note::
            For example plots see the :ref:`Obspy Gallery <gallery>`.

        :type filename: str, optional
        :param filename: Name of output file
        :type show_coverage: bool, optional
        :param show_coverage: Enable/disable second axes with representation of
                data coverage time intervals.
        :type show_percentiles: bool, optional
        :param show_percentiles: Enable/disable plotting of approximated
                percentiles. These are calculated from the binned histogram and
                are not the exact percentiles.
        :type show_histogram: bool, optional
        :param show_histogram: Enable/disable plotting of histogram. This
                can be set ``False`` e.g. to make a plot with only percentiles
                plotted. Defaults to ``True``.
        :type percentiles: list[int]
        :param percentiles: percentiles to show if plotting of percentiles is
                selected.
        :type show_noise_models: bool, optional
        :param show_noise_models: Enable/disable plotting of noise models.
        :type show_earthquakes: bool, optional
        :param show_earthquakes: Enable/disable plotting of earthquake models
            like in [ClintonHeaton2002]_ and [CauzziClinton2013]_. Disabled by
            default (``None``). Specify ranges (minimum and maximum) for
            magnitude and distance of earthquake models given as four floats,
            e.g. ``(0, 5, 0, 99)`` for magnitude 1.5 - 4.5 at a epicentral
            distance of 10 km. Note only 10, 100 and 3000 km distances and
            magnitudes 1.5 to 7.5 are available. Alternatively, a distance can
            be specified in last float of a tuple of three, e.g. ``(0, 5, 10)``
            for 10 km distance, or magnitude and distance can be specified in
            a tuple of two floats, e.g. ``(5.5, 10)`` for magnitude 5.5 at 10
            km distance.
        :type grid: bool, optional
        :param grid: Enable/disable grid in histogram plot.
        :type show: bool, optional
        :param show: Enable/disable immediately showing the plot. If
            ``show=False``, then the matplotlib figure handle is returned.
        :type max_percentage: float, optional
        :param max_percentage: Maximum percentage to adjust the colormap. The
            default is 30% unless ``cumulative=True``, in which case this value
            is ignored.
        :type period_lim: tuple(float, float), optional
        :param period_lim: Period limits to show in histogram. When setting
            ``xaxis_frequency=True``, this is expected to be frequency range in
            Hz.
        :type show_mode: bool, optional
        :param show_mode: Enable/disable plotting of mode psd values.
        :type show_mean: bool, optional
        :param show_mean: Enable/disable plotting of mean psd values.
        :type cmap: :class:`matplotlib.colors.Colormap`
        :param cmap: Colormap to use for the plot. To use the color map like in
            PQLX, [McNamara2004]_ use :class:`obspy.imaging.cm.pqlx`.
        :type cumulative: bool
        :param cumulative: Can be set to `True` to show a cumulative
            representation of the histogram, i.e. showing color coded for each
            frequency/amplitude bin at what percentage in time the value is
            not exceeded by the data (similar to the `percentile` option but
            continuously and color coded over the whole area). `max_percentage`
            is ignored when this option is specified.
        :type cumulative_number_of_colors: int
        :param cumulative_number_of_colors: Number of discrete color shades to
            use, `None` for a continuous colormap.
        :type xaxis_frequency: bool
        :param xaxis_frequency: If set to `True`, the x axis will be frequency
            in Hertz as opposed to the default of period in seconds.
        """
        import matplotlib.pyplot as plt
        self._PPSD__check_histogram()
        fig = plt.figure()
        fig.ppsd = AttribDict()

        if show_coverage:
            ax = fig.add_axes([0.12, 0.3, 0.90, 0.6])
            ax2 = fig.add_axes([0.15, 0.17, 0.7, 0.04])
        else:
            ax = fig.add_subplot(111)

        if show_percentiles:
            # for every period look up the approximate place of the percentiles
            for percentile in percentiles:
                periods, percentile_values = \
                    self.get_percentile(percentile=percentile)
                if xaxis_frequency:
                    xdata = 1.0 / periods
                else:
                    xdata = periods
                ax.plot(xdata, percentile_values, color="black", zorder=8)

        if show_mode:
            periods, mode_ = self.get_mode()
            if xaxis_frequency:
                xdata = 1.0 / periods
            else:
                xdata = periods
            if cmap.name == "viridis":
                color = "0.8"
            else:
                color = "black"
            ax.plot(xdata, mode_, color=color, zorder=9)

        if show_mean:
            periods, mean_ = self.get_mean()
            if xaxis_frequency:
                xdata = 1.0 / periods
            else:
                xdata = periods
            if cmap.name == "viridis":
                color = "0.8"
            else:
                color = "black"
            ax.plot(xdata, mean_, color=color, zorder=9)

        # Choose the correct noise model
        if self.special_handling == "infrasound":
            # Use IDC global infrasound models
            models = (get_idc_infra_hi_noise(), get_idc_infra_low_noise())

        elif self.special_handling == "accelerometer":
            # Use IDC global infrasound models
            models = (self.get_ahnm(), self.get_alnm())

        else:
            # Use Peterson NHNM and NLNM
            models = (get_nhnm(), get_nlnm())

        if show_noise_models:
            for periods, noise_model in models:
                if xaxis_frequency:
                    xdata = 1.0 / periods
                else:
                    xdata = periods
                ax.plot(xdata, noise_model, '0.4', linewidth=2, zorder=10)

        if show_earthquakes is not None:
            if len(show_earthquakes) == 2:
                show_earthquakes = (show_earthquakes[0],
                                    show_earthquakes[0] + 0.1,
                                    show_earthquakes[1],
                                    show_earthquakes[1] + 1)
            if len(show_earthquakes) == 3:
                show_earthquakes += (show_earthquakes[-1] + 1, )
            min_mag, max_mag, min_dist, max_dist = show_earthquakes
            for key, data in earthquake_models.items():
                magnitude, distance = key
                frequencies, accelerations = data
                accelerations = np.array(accelerations)
                frequencies = np.array(frequencies)
                periods = 1.0 / frequencies
                # Eq.1 from Clinton and Cauzzi (2013) converts
                # power to density
                ydata = accelerations / (periods ** (-.5))
                ydata = 20 * np.log10(ydata / 2)
                if not (min_mag <= magnitude <= max_mag and
                        min_dist <= distance <= max_dist and
                        min(ydata) < self.db_bin_edges[-1]):
                    continue
                xdata = periods
                if xaxis_frequency:
                    xdata = frequencies
                ax.plot(xdata, ydata, '0.4', linewidth=2)
                leftpoint = np.argsort(xdata)[0]
                if not ydata[leftpoint] < self.db_bin_edges[-1]:
                    continue
                ax.text(xdata[leftpoint],
                        ydata[leftpoint],
                        'M%.1f\n%dkm' % (magnitude, distance),
                        ha='right', va='top',
                        color='w', weight='bold', fontsize='x-small',
                        path_effects=[withStroke(linewidth=3,
                                                 foreground='0.4')])

        if show_histogram:
            label = "[%]"
            if cumulative:
                label = "non-exceedance (cumulative) [%]"
                if max_percentage is not None:
                    msg = ("Parameter 'max_percentage' is ignored when "
                           "'cumulative=True'.")
                    warnings.warn(msg)
                max_percentage = 100
                if cumulative_number_of_colors is not None:
                    cmap = LinearSegmentedColormap(
                        name=cmap.name, segmentdata=cmap._segmentdata,
                        N=cumulative_number_of_colors)
            elif max_percentage is None:
                # Set default only if cumulative is not True.
                max_percentage = 30

            fig.ppsd.cumulative = cumulative
            fig.ppsd.cmap = cmap
            fig.ppsd.label = label
            fig.ppsd.max_percentage = max_percentage
            fig.ppsd.grid = grid
            fig.ppsd.xaxis_frequency = xaxis_frequency
            if max_percentage is not None:
                color_limits = (0, max_percentage)
                fig.ppsd.color_limits = color_limits

            self._plot_histogram(fig=fig)

        if xaxis_frequency:
            ax.set_xlabel('Frequency [Hz]')
            ax.invert_xaxis()
        else:
            ax.set_xlabel('Period [s]')
        ax.set_xscale('log')
        ax.set_xlim(period_lim)
        ax.set_ylim(self.db_bin_edges[0], self.db_bin_edges[-1])
        if self.special_handling is None:
            ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
        elif self.special_handling == "infrasound":
            ax.set_ylabel('Amplitude [$Pa^2/Hz$] [dB]')
        else:
            ax.set_ylabel('Amplitude [dB]')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_title(self._get_plot_title())

        if show_coverage:
            self._PPSD__plot_coverage(ax2)
            # emulating fig.autofmt_xdate():
            for label in ax2.get_xticklabels():
                label.set_ha("right")
                label.set_rotation(30)

        # Catch underflow warnings due to plotting on log-scale.
        with np.errstate(all="ignore"):
            if filename is not None:
                plt.savefig(filename)
                plt.close()
            elif show:
                plt.draw()
                plt.show()
            else:
                plt.draw()
                return fig
