"""
Name: prob_run.py
Authors: Stephan Meighen-Berger
Generates simulation runs using the probabilistic model
"""

# General imports
import numpy as np
import pandas as pd
import sys
import pickle
import time
from scipy.signal import find_peaks
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
import logging
from pdfs import construct_pdf

def main(run_count):
    start = time.time()
    storage_loc = '/home/ga78fed/output/'
    logging.basicConfig(filename='/home/ga78fed/output/prob.log', level=logging.DEBUG)
    # Parameters
    logging.debug("Setting parameteres")
    detector_position = np.array([2., 0.])
    dens = 1e0
    acceptance_range = np.array([30., 90.])
    simulation_step = 0.1
    simulation_time = 5000.
    run_counts = 1
    wavelengths = np.linspace(300., 600., 301)
    emission_time = 100.
    photon_counts_1 = 1e10
    photon_counts_2 = 5e9
    photon_counts_3 = 1e11
    efficiency = 0.1
    water_vel = 0.05 * simulation_step
    rest_time = 100. / simulation_step
    species = np.array(["Species 1", "Species 2", "Species 3"])
    gamma_test = construct_pdf(
        {"class": "Gamma",
        "mean": 0.5 / simulation_step,
        "sd": 0.45 / simulation_step
        })
    gamma_test_2 = construct_pdf(
        {"class": "Gamma",
        "mean": 0.2 / simulation_step,
        "sd": 0.15 / simulation_step
        })
    gamma_test_3 = construct_pdf(
        {"class": "Gamma",
         "mean": 1. / simulation_step,
         "sd": 0.85 / simulation_step
        }
    )
    gauss_test = construct_pdf(
        {"class": "Normal",
        "mean": 450.,
        "sd": 50.
        })
    min_y = 0.
    max_y = 3.
    max_x = 26.
    starting_pop = 2
    pop_size = starting_pop
    injection_count = dens * (max_y - min_y) * water_vel
    expected_counts = int(injection_count * simulation_time)
    # Normalizing pdfs
    logging.debug("Normalizing pdfs")
    norm_time_series_1 = (
        gamma_test.pdf(np.arange(0., emission_time, simulation_step)) /
        np.trapz(gamma_test.pdf(np.arange(0., emission_time, simulation_step)),
                np.arange(0., emission_time, simulation_step))
    )
    norm_time_series_2 = (
        gamma_test_2.pdf(np.arange(0., emission_time, simulation_step)) /
        np.trapz(gamma_test_2.pdf(np.arange(0., emission_time, simulation_step)),
                np.arange(0., emission_time, simulation_step))
    )
    norm_time_series_3 = (
        gamma_test_3.pdf(np.arange(0., emission_time, simulation_step)) /
        np.trapz(gamma_test_3.pdf(np.arange(0., emission_time, simulation_step)),
                 np.arange(0., emission_time, simulation_step))
    )
    norm_time_series_1 = norm_time_series_1 * photon_counts_1
    norm_time_series_2 = norm_time_series_2 * photon_counts_2
    norm_time_series_3 = norm_time_series_3 * photon_counts_3
    norm_dic = {
        species[0]: norm_time_series_1,
        species[1]: norm_time_series_2,
        species[2]: norm_time_series_3
    }
    norm_wavelengths = (
        gauss_test.pdf(wavelengths) /
        np.trapz(gauss_test.pdf(wavelengths), wavelengths)
    )
    # The attenuation function
    logging.debug("Setting the attenuation function")
    attenuation_vals = np.array([
        [
            299.,
            329.14438502673795, 344.11764705882354, 362.2994652406417,
            399.44415494181, 412.07970421102266, 425.75250006203635,
            442.53703565845314, 457.1974490682151, 471.8380108687561,
            484.3544504826423, 495.7939402962853, 509.29799746891985,
            519.6903148961513, 530.0627807141617, 541.5022705278046,
            553.9690811186382, 567.4929899004939, 580.9771954639073,
            587.1609717362714, 593.3348222040249, 599.4391920395047,
            602.4715253480235
        ],
        [
            0.8,
            0.6279453220864465,0.3145701363176568,
            0.12591648888305143,0.026410321551339357, 0.023168667048510762,
            0.020703255370450736, 0.019552708373076478,
            0.019526153330089138, 0.020236306473695613,
            0.02217620815962483, 0.025694647290888873,
            0.031468126242251794, 0.03646434475343956,
            0.04385011375530569, 0.05080729755501162,
            0.061086337538657706, 0.07208875589035815, 0.09162216168767365,
            0.11022281058708046, 0.1350811713674855, 0.18848851206491904,
            0.23106528395398912
        ]
    ])
    atten_spl = UnivariateSpline(attenuation_vals[0],
        attenuation_vals[1], k=1, s=0)
    atten_vals = atten_spl(wavelengths)
    # Loading prob model
    logging.debug("Loading the prob model")
    spl_prob = load_and_parse('/home/ga78fed/projects/prob_model/prob_model/')
    logging.debug("The true values")
    number_of_peaks_base, peak_heights_base, peak_widths_base = (
        evaluation(run_counts, water_vel, simulation_time, simulation_step,
            expected_counts, atten_vals, starting_pop, min_y, max_y,
            max_x, species,
            spl_prob, wavelengths, detector_position, acceptance_range,
            norm_wavelengths, efficiency, norm_dic, emission_time,
            pop_size, rest_time, id_wave=[50, 100, 150, 200, 250, 300])
    )
    logging.debug("Dumping true values")
    pickle.dump([number_of_peaks_base, peak_heights_base, peak_widths_base],
                open(storage_loc + "results/base_v4_1_sim_%d.p" %run_count, "wb" ) )
    # The stuff to compare to
    logging.debug("Comparison values")
    res_dic = {}
    factors_arr = [1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1,
                   6e-1, 7e-1, 8e-1, 9e-1]
    for factors in factors_arr:
        atten_copy = np.copy(atten_vals)
        atten_copy = atten_copy * factors
        number_of_peaks, peak_heights, peak_widths = evaluation(
            run_counts, water_vel, simulation_time, simulation_step,
            expected_counts, atten_copy, starting_pop, min_y, max_y,
            max_x, species,
            spl_prob, wavelengths, detector_position, acceptance_range,
            norm_wavelengths, efficiency, norm_dic, emission_time,
            pop_size, rest_time, id_wave=[50, 100, 150, 200, 250, 300])
        res_dic[factors] = [number_of_peaks, peak_heights, peak_widths]
    # Storing
    logging.debug("Dumping comparison values")
    pickle.dump(res_dic, open(storage_loc + "results/comp_v4_1_sim_%d.p" %run_count, "wb" ))
    end = time.time()
    print("Execution time:")
    print(end-start)

def load_and_parse(storage_loc):
    data_0 = pickle.load(open(storage_loc + "offcenter_v2_0.p", "rb"))
    data_1 = pickle.load(open(storage_loc + "offcenter_v2_1.p", "rb"))
    data_2 = pickle.load(open(storage_loc + "offcenter_v2_2.p", "rb"))
    data_3 = pickle.load(open(storage_loc + "offcenter_v2_3.p", "rb"))
    data_5 = pickle.load(open(storage_loc + "offcenter_v2_5.p", "rb"))
    id_alpha = 0
    counts_0, edges_0 = np.histogram(
        data_0['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_0['norm'][id_alpha]
    )
    counts_1, _ = np.histogram(
        data_1['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_1['norm'][id_alpha]
    )
    counts_2, _ = np.histogram(
        data_2['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_2['norm'][id_alpha]
    )
    counts_3, _ = np.histogram(
        data_3['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_3['norm'][id_alpha]
    )
    counts_5, _ = np.histogram(
        data_5['x_arr'][id_alpha],
        bins=np.linspace(0., 26., 131),
        weights=1./data_5['norm'][id_alpha]
    )
    spl_prob = RectBivariateSpline(
        (edges_0[1:] + edges_0[:-1]) / 2.,
        np.array([0., 1., 2., 3., 5.]),
        np.array([counts_0, counts_1,
                  counts_2, counts_3, counts_5]).T, s=0.4)
    return spl_prob

def run_func(water_vel, simulation_time, simulation_step, expected_counts,
             atten_func, starting_pop, min_y, max_y, max_x, species,
             spl_prob, wavelengths, detector_position,
             acceptance_range, norm_wavelengths, efficiency,
             norm_dic, emission_time, pop_size, rest_time, id_wave=[200]):
    logging.debug("Launching new simulation run!")
    injection_times = (
        np.sort(np.random.randint(
            int(simulation_time / simulation_step),
            size=expected_counts))
    )
    unqiue_times, unique_counts = (
        np.unique(injection_times, return_counts=True)
    )
    # The population
    population = pd.DataFrame(
        {
            "species": None,
            "pos_x": 0.,
            "pos_y": 0.,
            "observed": True,
            "flashing": False,
            "can_flash": True,
            "rest_time": 0,
        },
        index=np.arange(starting_pop),
    )
    population.loc[:, 'pos_y'] = np.random.uniform(min_y, max_y, starting_pop)
    # Species
    if len(species) > 1:
        pop_index_sample = np.random.randint(
            0, len(species), starting_pop
        )
    elif len(species) == 1:
        pop_index_sample = np.zeros(starting_pop, dtype=np.int)
    population.loc[:, 'species'] = (
        species[pop_index_sample]
    )
    statistics = list(range(int(simulation_time / simulation_step)))
    for i in range(int(simulation_time / simulation_step)):
        counter = 0
        # Resetting the flash
        population.loc[:, 'flashing'] = False
        if i in unqiue_times:
            inject = unique_counts[counter]
            for j in range(inject):
                if len(species) > 1:
                    pop_index_sample = np.random.randint(
                        0, len(species), 1
                    )
                elif len(species) == 1:
                    pop_index_sample = np.zeros(1, dtype=np.int)
                population.loc[pop_size + (j+1)] = [
                    species[pop_index_sample][0],
                    0.,
                    np.random.uniform(min_y, max_y),
                    True,
                    False,
                    True,
                    0
                ]
                pop_size += 1
            counter += 1
        # Injection only according to array
        observation_mask = population.loc[:, 'observed']
        # propagation
        population.loc[observation_mask, 'pos_x'] = (
            population.loc[observation_mask, 'pos_x'] + water_vel
        )
        # Checking if should emit
        prob_arr = spl_prob(
            population.loc[observation_mask, 'pos_x'].values,
            population.loc[observation_mask, 'pos_y'].values, grid=False)
        prob_arr[prob_arr < 0.] = 0.
        flash_mask = np.logical_and(
            np.array(np.random.binomial(1, prob_arr, len(prob_arr)),
                     dtype=bool),
            population.loc[observation_mask, 'can_flash'].values)
        tmp_flash_mask = population.loc[observation_mask, 'flashing'].values
        tmp_flash_mask += flash_mask
        population.loc[observation_mask, 'flashing'] = tmp_flash_mask
        # population.loc[observation_mask, 'flashing'] += flash_mask
        can_flash_mask =  population.loc[:, 'flashing'].values
        population.loc[can_flash_mask, 'can_flash'] = False
        # Counting the rest
        resting_mask = population.loc[:, 'can_flash'].values
        population.loc[~resting_mask, 'rest_time'] += 1
        # Checking if can flash again
        flash_mask = np.greater(population.loc[:, 'rest_time'], rest_time)
        population.loc[flash_mask, 'rest_time'] = 0
        population.loc[flash_mask, 'can_flash'] = True
        # Observed
        new_observation_mask = np.less(
            population.loc[observation_mask, 'pos_x'], max_x
        )
        population.loc[observation_mask, 'observed'] = new_observation_mask
        statistics[i] = population.copy()
    # Applying emission pdf
    # And propagating
    arriving_light = np.zeros(
        (int(simulation_time / simulation_step), len(wavelengths))
    )
    for id_step, pop in enumerate(statistics):
        flashing_mask = pop.loc[:, 'flashing'].values
        if np.sum(flashing_mask) > 0:
            x_arr = pop.loc[flashing_mask, 'pos_x'].values
            y_arr = pop.loc[flashing_mask, 'pos_y'].values
            species_arr = pop.loc[flashing_mask, "species"].values
            distances = np.sqrt(
                (x_arr - detector_position[0])**2. +
                (y_arr - detector_position[1])**2.
            )
            angles = np.array(
                np.arctan2(
                    (y_arr - detector_position[1]),
                    (x_arr - detector_position[0]))
            )
            angles = np.degrees(angles)
            outside_minus = np.less(angles, acceptance_range[0])
            outside_plus = np.greater(angles, acceptance_range[1])
            angle_check = np.logical_and(~outside_minus, ~outside_plus)
            angle_squash = angle_check.astype(float)
            atten_facs = np.array([
                np.exp(-distances[id_flash] * atten_func) /
                (4. * np.pi * distances[id_flash]**2.)
                for id_flash in range(np.sum(flashing_mask))
            ])
            curr_pulse = np.array([
                [
                    (norm_time * norm_wavelengths * atten_facs[id_flash]) *
                     efficiency * angle_squash[id_flash]
                    for norm_time in norm_dic[species_arr[id_flash]]
                ]
                for id_flash in range(np.sum(flashing_mask))
            ])
            # Checking if end is being overshot
            if (id_step +
                int(emission_time / simulation_step) <= len(arriving_light)):
                arriving_light[id_step:id_step+
                    int(emission_time / simulation_step), :] += (
                    np.sum(curr_pulse, axis=0)
                )
            else:
                arriving_light[id_step:id_step+
                    int(emission_time / simulation_step), :] += (
                    np.sum(curr_pulse, axis=0)[0:len(arriving_light) -
                        (id_step+int(emission_time / simulation_step))]
                )
    print(arriving_light)
    data_test = np.array([
            arriving_light[:, id_w] for id_w in id_wave
    ])
    x_grid = np.arange(0., simulation_time, simulation_step)
    results_p = []
    results_prop = []
    print(data_test)
    for id_w, _ in enumerate(id_wave):
            peaks, properties = find_peaks(data_test[id_w], prominence=1, width=20)
            results_p.append(peaks)
            results_prop.append(properties)
    number_of_peaks = np.array([
        len(peaks) for peaks in results_p
    ])
    heights = np.array([
        properties["prominences"] for properties in results_prop
    ])
    print(heights)
    widths = np.array([
        x_grid[properties["right_ips"].astype(int)] -
        x_grid[properties["left_ips"].astype(int)] for properties in results_prop
    ])
    return [
        number_of_peaks,
        heights,
        widths
    ]

def evaluation(number_of_runs, water_vel, simulation_time, simulation_step,
    expected_counts, atten_func, starting_pop, min_y, max_y, max_x, species,
    spl_prob, wavelengths, detector_position, acceptance_range,
    norm_wavelengths, efficiency, norm_dic, emission_time,
    pop_size, rest_time, id_wave=[200]):
    logging.debug("----------------------------------------------------------------")
    logging.debug("New simulation set")
    pop_size = starting_pop
    res = []
    def map_func(counter):
        return run_func(
            water_vel, simulation_time, simulation_step,
            expected_counts, atten_func, starting_pop, min_y, max_y,
            max_x, species,
            spl_prob, wavelengths, detector_position, acceptance_range,
            norm_wavelengths, efficiency, norm_dic, emission_time,
            pop_size, rest_time, id_wave)
    run_list = list(range(number_of_runs))
    for count in run_list:
            res.append(map_func(count))
    ret_peaks = np.array([r[0] for r in res])
    ret_heights = np.array([r[1] for r in res])
    ret_widths = np.array([r[2] for r in res])
    # res = np.array([map_func(count) for count in run_list], dtype=object)
    logging.debug("Finished simulation set")
    logging.debug("-----------------------------------------------------------------")
    return ret_peaks, ret_heights, ret_widths

if __name__ == "__main__":
    main(int(sys.argv[1]))
