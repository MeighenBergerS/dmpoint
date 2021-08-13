# Name: dm_source_batch.py
# Authors: Stephan Meighen-Berger
# Runs a simplified dm point source test

# General imports
# Imports
import numpy as np
import pickle
import csv
import sys
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
import time
import logging

def main(population):
    start = time.time()
    storage_loc = '/home/ga78fed/output/'  # Needs to be changed by the user
    file_tree = '/home/ga78fed/projects/dmpoint/'  # Point to the module
    logging.basicConfig(
        filename='/home/ga78fed/output/prob.log', level=logging.DEBUG
    )
    logging.debug("Welcome to dmpoint! We'll be testing some dm today")
    # Parameters
    logging.debug("-----------------------------------------------------------")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Setting parameteres")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Setting constants")
    # Constants
    seconds = 60.
    minutes = 60.
    days = seconds * minutes * 24
    m_eborders = np.logspace(4., 8., 41)
    m_ewidths = np.diff(m_eborders)
    m_egrid = np.sqrt(m_eborders[1:]*m_eborders[:-1])
    minimal_resolution = 0.2
    ra_grid = np.arange(0., 360., minimal_resolution)
    decl_grid = np.arange(0., 10., minimal_resolution)
    flux_scan = np.logspace(-24, -14, 51)  # Fluxes to test
    ide_scan = [0, 10, 20, 30]  # energy ids to test
    signal_test = 100  # Number of signal samples to use
    logging.debug("Finished the constants")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Fetching the effective areas")
    eff_areas = [
        file_tree + 'data/icecube_10year_ps/irfs/IC40_effectiveArea.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC59_effectiveArea.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC79_effectiveArea.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_I_effectiveArea.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_II_effectiveArea.csv',
    ]
    eff_dic = {
        0: ice_parser(eff_areas[0]),
        1: ice_parser(eff_areas[1]),
        2: ice_parser(eff_areas[2]),
        3: ice_parser(eff_areas[3]),
        4: ice_parser(eff_areas[4]),
        5: ice_parser(eff_areas[4]),
        6: ice_parser(eff_areas[4]),
        7: ice_parser(eff_areas[4]),
        8: ice_parser(eff_areas[4]),
        9: ice_parser(eff_areas[4]),
    }
    logging.debug("Finished loading the effective areas")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Fetching the uptimes")
    # MJD, log10(E/GeV), AngErr[deg], RA[deg], Dec[deg], Azimuth[deg], Zenith[deg]
    uptime_sets = [
        file_tree + 'data/icecube_10year_ps/uptime/IC40_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC59_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC79_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC86_I_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC86_II_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC86_III_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC86_IV_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC86_V_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC86_VI_exp.csv',
        file_tree + 'data/icecube_10year_ps/uptime/IC86_VII_exp.csv',
    ]
    uptime_dic = {
        0: ice_parser(uptime_sets[0]),
        1: ice_parser(uptime_sets[1]),
        2: ice_parser(uptime_sets[2]),
        3: ice_parser(uptime_sets[3]),
        4: ice_parser(uptime_sets[4]),
        5: ice_parser(uptime_sets[5]),
        6: ice_parser(uptime_sets[6]),
        7: ice_parser(uptime_sets[7]),
        8: ice_parser(uptime_sets[8]),
        9: ice_parser(uptime_sets[9]),
    }
    uptime_tot_dic = {}
    for year in range(10):
        uptime_tot_dic[year] = np.sum(np.diff(uptime_dic[year])) * days
    logging.debug("Finished loading the uptimes")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Fetching the smearing matrices")
    # Loading smearing
    # log10(E_nu/GeV)_min, log10(E_nu/GeV)_max, Dec_nu_min[deg], Dec_nu_max[deg], log10(E/GeV), PSF_min[deg], PSF_max[deg],
    # AngErr_min[deg], AngErr_max[deg], Fractional_Counts
    smearing_sets = [
        file_tree + 'data/icecube_10year_ps/irfs/IC40_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC59_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC79_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_I_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
        file_tree + 'data/icecube_10year_ps/irfs/IC86_II_smearing.csv',
    ]
    smearing_dic = {
        0: ice_parser(smearing_sets[0]),
        1: ice_parser(smearing_sets[1]),
        2: ice_parser(smearing_sets[2]),
        3: ice_parser(smearing_sets[3]),
        4: ice_parser(smearing_sets[4]),
        5: ice_parser(smearing_sets[5]),
        6: ice_parser(smearing_sets[6]),
        7: ice_parser(smearing_sets[7]),
        8: ice_parser(smearing_sets[8]),
        9: ice_parser(smearing_sets[9]),
    }
    logging.debug("Finished loading the smearing matrices")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Loading the event weights")
    weights = weight_constructor(
        file_tree + 'data/simulated_data_bkgrd_store_benchmark.pkl'
    )
    logging.debug("Finished loading the event weights")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Loading the background")
    bkgrd_data = pickle.load(
        open(file_tree + 'data/test_set.pkl', "rb")
    )
    q_bkgrd = bkgrd_data[0]
    dens_bkgrd = bkgrd_data[1]
    logging.debug("Finished loading background")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Finished the parameter loading")
    logging.debug("-----------------------------------------------------------")
    logging.debug("-----------------------------------------------------------")
    logging.debug("Starting the calculation loop")
    cl_lim_store = []
    pop = population
    for ide in tqdm(ide_scan):
        cl_tmp = []
        for flux in flux_scan:
            signal_densities = np.array([
                signal_density_constructor(
                    ide, pop, flux, seed,
                    m_egrid, m_ewidths, eff_dic,
                    uptime_tot_dic, smearing_dic, weights,
                    decl_grid, ra_grid, angle_uncer=1.
                ) for seed in range(signal_test)
            ])
            cl_lim = (
                improved_comparison_signal(
                    signal_densities, q_bkgrd, dens_bkgrd
                )
            )
            cl_tmp.append(cl_lim)
        cl_lim_store.append(cl_tmp)
    logging.debug("Finished the calculation loop")
    end = time.time()
    print("Execution time:")
    print(end-start)
    logging.debug("-----------------------------------------------------------")
    logging.debug("-----------------------------------------------------------")
    pickle.dump(
        cl_lim_store,
        open(storage_loc + "results/cl_lim_res_pop_%d.p" %population, "wb" )
    )
    # Storing
    logging.debug("Dumping values")
    logging.debug("Finished, have a good day!")
    logging.debug("-----------------------------------------------------------")
    logging.debug("-----------------------------------------------------------")
    

def ice_parser(filename):
    """ Helper function to parse icecube data

    Parameters
    ----------
    filename : str
        The file to parse

    Returns
    -------
    store : np.array
        The parsed data
    """
    store = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row_num, row in enumerate(reader):
            if row_num == 0:
                continue
            store.append(row[0].split())
    store = np.array(store, dtype=float)
    return store

def smearing_function(true_e, true_dec, year, smearing_dic):
    """ Helper function to smeare events according to the smearing matrices

    Parameters
    ----------
    true_e : float
        The neutrino energy as log_10 (E/GeV)
    true_dec : float
        The true neutrino declination
    year : int
        The year the event was recorded
    smearing_dic : dic
        The smearing matrices

    Returns
    -------
    smearing_e_grid : np.array
        The smeared energy grid
    smearing_fraction : np.array
        The fractional weights for each smearing energy grid bin
    """
    e_test = true_e
    angle_test = true_dec
    local_smearing = smearing_dic[year]
    cross_check_smear_egrid = (local_smearing[:, 1] + local_smearing[:, 0])/2.
    idE = np.abs(cross_check_smear_egrid - e_test).argmin()
    all_near_e = (
        np.where(cross_check_smear_egrid == cross_check_smear_egrid[idE])[0]
    )
    cross_check_smear_theta = (local_smearing[:, 2] + local_smearing[:, 3])/2.
    idtheta = np.abs(cross_check_smear_theta - angle_test).argmin()
    all_near_theta = (
        np.where(cross_check_smear_theta == cross_check_smear_theta[idtheta])[0]
    )
    elements_of_interest = np.intersect1d(all_near_e, all_near_theta)
    tmp_local_smearing = local_smearing[elements_of_interest]
    smearing_e_grid = np.unique(tmp_local_smearing[:, 4])
    smearing_fraction = []
    for smearing_e_loop in smearing_e_grid:
        idE = np.abs(tmp_local_smearing[:, 4] - smearing_e_loop).argmin()
        all_near_e = (
            np.where(
                tmp_local_smearing[:, 4] == tmp_local_smearing[:, 4][idE]
            )[0]
        )
        smearing_fraction.append(np.sum(tmp_local_smearing[all_near_e][:, -1]))
    # Normalizing
    smearing_fraction = (
        np.array(smearing_fraction) /
        np.trapz(smearing_fraction, x=smearing_e_grid)
    )
    return smearing_e_grid, smearing_fraction

def effective_area_func(injection_id, theta, year,
                        m_egrid, m_ewidths, eff_dic, uptime_tot_dic):
    """ Helper function to apply the effective area and other normalizations

    Parameters
    ----------
    injection_id : int
        The energy index used for the event according to m_egrid
    theta : float
        The injection angle
    year : int
        The year the event was recorded
    m_egrid : np.array
        The energy grid used for the signal
    m_ewidths : np.array
        The energy grid bin widths used for the signal
    eff_dic : dic
        The effective area dictionary
    uptime_tot_dic: dic 
        The detector uptime dic

    Returns
    -------
    unsmeared_counts : np.array
        The still to be smeared expected reconstructed muon counts
    """
    cross_check_egrid = (eff_dic[year][:, 1] + eff_dic[year][:, 0])/2.
    cross_check_theta = (eff_dic[year][:, 2] + eff_dic[year][:, 3])/2.
    eff_areas = []
    check_angle = (theta)
    energy = m_egrid[injection_id]
    loge = np.log10(energy)
    idE = np.abs(cross_check_egrid - loge).argmin()
    all_near = (np.where(cross_check_egrid == cross_check_egrid[idE])[0])
    idTheta = np.abs(cross_check_theta[all_near] - check_angle).argmin()
    eff_areas.append(eff_dic[year][all_near, -1][idTheta])
    loc_eff_area = np.array(eff_areas)
    unsmeared_counts = (
        m_ewidths[injection_id] * loc_eff_area *
        uptime_tot_dic[year] * 2. * np.pi
    )
    return unsmeared_counts

def sim_to_dec(injection_id, theta, year,
               m_egrid, m_ewidths, eff_dic, uptime_tot_dic, smearing_dic):
    """ Helper function to convert an injected event to reconstructed ones

    Parameters
    ----------
    injection_id : int
        The energy index used for the event according to m_egrid
    theta : float
        The injection angle
    year : int
        The year the event was recorded
    m_egrid : np.array
        The energy grid used for the signal
    m_ewidths : np.array
        The energy grid bin widths used for the signal
    eff_dic : dic
        The effective area dictionary
    uptime_tot_dic: dic 
        The detector uptime dic
    smearing_dic : dic
        The smearing matrices

    Returns
    -------
    smeared : np.array
        The smeared reconstructed events
    """
    # Converts simulation data to detector data
    unnormalized_counts = effective_area_func(
        injection_id, theta, year, m_egrid, m_ewidths, eff_dic, uptime_tot_dic
    )
    log_egrid = np.log10(m_egrid)
    smeared = []
    check_angle = (theta)
    smearing_e, smearing = smearing_function(
        log_egrid[injection_id],
        check_angle, year,
        smearing_dic
    )
    spl = UnivariateSpline(smearing_e, smearing * unnormalized_counts,
                            k=1, s=0, ext=1)
    smeared = spl(log_egrid)
    return smeared

# Loading weights
def weight_constructor(path_to_file: str) -> UnivariateSpline:
    """ constructes the energy weights for the IceCube events

    Parameters
    ----------
    path_to_file : str
        The theoretical preidcitons for the counts

    Returns
    -------
    weights : UnivariateSpline
        Spline which returns the weight corresponding to the input energy
        as log_10(E/GeV)
    """
    theoretical_predictions = pickle.load(open(path_to_file, "rb"))
    egrid = theoretical_predictions[0]
    weights_arr = np.nan_to_num(theoretical_predictions[2] /
                                theoretical_predictions[3])
    # weights_arr[weights_arr > 1] = 1
    return UnivariateSpline(egrid, weights_arr, k=1, s=0)

# Construct weighted events
def weighted_events(event_dic, weight_func, years):
    """ Weights the events and transforms for later usage

    Parameters
    ----------
    event_dic : dic
        The events to weight
    weight_func : function
        The weighting function (usually a spline)
    years : list-like
        The years of interest

    Returns
    -------
    np.array
        All weighted events as a numpy array
    """
    weighted_data = []
    for year in years:
        weighted_data.append(np.array([
            event_dic[year][:, 3],
            event_dic[year][:, 4],
            weight_func(event_dic[year][:, 1]),
            event_dic[year][:, 2]
        ]))
        weighted_data[-1] = weighted_data[-1].T
    return np.concatenate([weighted_data[year] for year in years])

# Signal density constructor
def signal_density_constructor(energy_id, number_of_sources,
                               flux_norm, seed, m_egrid, m_ewidths, eff_dic,
                               uptime_tot_dic, smearing_dic,
                               weights_func,
                               decl_grid, ra_grid, angle_uncer=0.5):
    """ Constructs a density profile for the desired signal

    Parameters
    ----------
    energy_id : int
        The energy index of the signal according to m_egrid
    number_of_sources : int
        The number of sources to sample
    flux_norm : float
        The flux normalization
    seed : int
        The seed to use in the calculation
    m_egrid : np.array
        The energy grid used for the signal
    m_ewidths : np.array
        The energy grid bin widths used for the signal
    eff_dic : dic
        The effective area dictionary
    uptime_tot_dic: dic 
        The detector uptime dic
    smearing_dic : dic
        The smearing matrices
    weights_func : function
        The weighting function (usually a spline)
    decl_grid : np.array
        The declination grid
    ra_grid : np.array
        The right ascension grid
    angle_uncertainty : float
        The angular uncertainty used for the signal events

    Returns
    -------
    np.array
        The sampled binned density grid for the signal
    """
    rand_state = np.random.RandomState(seed)
    declination_sample = rand_state.uniform(
        0., 10., size=(number_of_sources)
    )
    ra_sample = rand_state.uniform(
        0., 360., size=(number_of_sources)
    )
    years = rand_state.randint(0, 9, size=(number_of_sources))
    count_distros = np.array([
        sim_to_dec(energy_id, declination_sample[ids], years[ids],
            m_egrid, m_ewidths, eff_dic, uptime_tot_dic, smearing_dic
        ) * flux_norm
        for ids in range(number_of_sources)
    ])
    events = np.array([[
        rand_state.poisson(counts) for counts in count_distro
    ] for count_distro in count_distros])
    reweighted_events = np.array([
        set_events * weights_func(np.log10(m_egrid))
        for set_events in events
    ])
    density_grid = np.zeros((len(decl_grid), len(ra_grid)))
    # Binning
    for set_id in range(number_of_sources):
        for event in reweighted_events[set_id]:
            # Finding grid points to add to
            # RA
            single_ra = False
            idra_low = (
                np.abs(ra_grid - (ra_sample[set_id] - angle_uncer))
            ).argmin()
            idra_high = (
                np.abs(ra_grid - (ra_sample[set_id] + angle_uncer))
            ).argmin()
            if idra_high == idra_low:
                single_ra = True
            # Declination
            single_dec = False
            iddec_low = (
                np.abs(decl_grid - (declination_sample[set_id] - angle_uncer))
            ).argmin()
            iddec_high = (
                np.abs(decl_grid - (declination_sample[set_id] + angle_uncer))
            ).argmin()
            if iddec_low == iddec_high:
                single_dec = True
            # Binning
            if (not single_ra) and (not single_dec):
                density_grid[iddec_low:iddec_high, idra_low:idra_high] += (
                    event
                )
            elif single_ra and (not single_dec):
                density_grid[iddec_low:iddec_high, idra_low] += (
                    event
                )
            elif (not single_ra) and single_dec:
                density_grid[iddec_low, idra_low:idra_high] += (
                    event
                )
            else:
                density_grid[iddec_low, idra_low] += (
                    event
                )
    return density_grid

def improved_comparison_signal(signals, q_bkgrd, bkgrd_density):
    """ Calculates the confidence limit for the signals

    Parameters
    ----------
    signals : np.array
        Array of density arrays for the signal
    q_bkgrd : float
        The test statistic to test against
    bgrd_density : np.array
        The background density

    Returns
    -------
    fraction : float
        The fraction passing the test, this is the confidence limit
    """
    chi_data = []
    for signal in signals:
        chi_data.append(np.sum(np.nan_to_num(
            (signal)**2. / bkgrd_density
        )))
    chi_data = np.array(chi_data)
    subset = chi_data[chi_data > q_bkgrd]
    fraction = len(subset) / len(chi_data)
    return fraction

if __name__ == "__main__":
    main(int(sys.argv[1]))