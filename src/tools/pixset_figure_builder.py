import pickle
import os
import numpy as np
from matplotlib import pyplot as plt


def sort_dict(item: dict):
    return {k: sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items())}


if __name__ == '__main__':
    category_counters = {}

    with open('../train_category_occurrences.pkl', 'rb') as f:
        category_counters['train'] = pickle.load(f)
    with open('../val_category_occurrences.pkl', 'rb') as f:
        category_counters['val'] = pickle.load(f)
    with open('../test_category_occurrences.pkl', 'rb') as f:
        category_counters['test'] = pickle.load(f)

    N = len(list(category_counters['train'].keys()))
    ind = np.arange(N)
    width = 0.3

    category_counters['train'] = sort_dict(category_counters['train'])
    category_counters['val'] = sort_dict(category_counters['val'])
    category_counters['test'] = sort_dict(category_counters['test'])


    _, ax = plt.subplots()
    train_occurrence_ratios = np.asarray(list(category_counters['train'].values()))/np.sum(np.asarray(list(category_counters['train'].values())))
    val_occurrence_ratios = np.asarray(list(category_counters['val'].values()))/np.sum(np.asarray(list(category_counters['val'].values())))
    test_occurrence_ratios = np.asarray(list(category_counters['test'].values()))/np.sum(np.asarray(list(category_counters['test'].values())))
    train_bars = ax.bar(ind,train_occurrence_ratios, width, color='r')
    val_bars = ax.bar(ind + width, val_occurrence_ratios, width, color='b')
    test_bars = ax.bar(ind + 2*width, test_occurrence_ratios, width, color='g')

    ax.set_ylabel('Occurrence ratios')
    ax.set_title('Category occurrence ratios')
    ax.set_xticks(ind + width / 3)
    ax.set_xticklabels(category_counters['train'].keys(), rotation=20)

    ax.legend((train_bars[0], val_bars[0], test_bars[0]), ('Train', 'Val', 'Test'))
    plt.savefig('category_occurrence_ratios.png')
    plt.show()

    _, ax = plt.subplots()
    val_occurrence_ratios = np.asarray(list(category_counters['val'].values()))/np.sum(np.asarray(list(category_counters['val'].values())))
    test_occurrence_ratios = np.asarray(list(category_counters['test'].values()))/np.sum(np.asarray(list(category_counters['test'].values())))
    train_bars = ax.bar(ind, category_counters['train'].values(), width, color='r')
    val_bars = ax.bar(ind + width, category_counters['val'].values(), width, color='b')
    test_bars = ax.bar(ind + 2*width, category_counters['test'].values(), width, color='g')

    ax.set_ylabel('Occurrences')
    ax.set_title('Category occurrences')
    ax.set_xticks(ind + width / 3)
    ax.set_xticklabels(category_counters['train'].keys(), rotation=20)

    ax.legend((train_bars[0], val_bars[0], test_bars[0]), ('Train', 'Val', 'Test'))
    plt.savefig('category_occurrence.png')
    plt.show()

    dataset_info = {
        '20200706_171559_part27_1170_1370': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_180421_part41_1800_2500': {
            'location': 'highway', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_162218_part21_4368_7230': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_144800_part25_1224_2100': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200803_151243_part45_4780_5005': {
            'location': 'boulevard', 'time': 'day', 'weather': 'rain'
        },
        '20200721_181359_part42_1903_2302': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200730_003948_part44_5818_6095': {
            'location': 'downtown', 'time': 'night', 'weather': 'no rain'
        },
        '20200706_170136_part28_2060_2270': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_202209_part31_2980_3091': {
            'location': 'suburban', 'time': 'day', 'weather': 'no rain'
        },
        '20200618_184930_part16_3030_3200': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200618_175654_part15_1380_1905': {
            'location': 'parking_lot', 'time': 'day', 'weather': 'no rain'
        },
        '20200616_145121_part7_2575_2860': {
            'location': 'suburban', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_162218_part21_790_960': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_202209_part31_2636_2746': {
            'location': 'suburban', 'time': 'day', 'weather': 'no rain'
        },
        '20200803_174859_part46_2761_2861': {
            'location': 'boulevard', 'time': 'day', 'weather': 'rain'
        },
        '20200706_162218_part21_4070_4170': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200617_195023_part14_4707_4850': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_164103_part43_2361_2481': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_143808_part26_3042_3420': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200611_184008_part3_3130_3290': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_143808_part26_2370_2500': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200708_121622_part33_5088_5209': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200617_191627_part12_1614_1842': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200617_195023_part14_1872_2050': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_144638_part36_1956_2229': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200618_184930_part16_4191_4420': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200730_003948_part44_275_550': {
            'location': 'boulevard', 'time': 'night', 'weather': 'no rain'
        },
        '20200721_165008_part39_1_220': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_143404_part35_4400_4608': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200611_172353_part5_150_250': {
            'location': 'parking_lot', 'time': 'day', 'weather': 'rain'
        },
        '20200721_155900_part38_549_953': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_151313_part23_2880_3120': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_143404_part35_3268_3389': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_154835_part37_696_813': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200708_121622_part33_5534_5833': {
            'location': 'boulevard', 'time': 'day', 'weather': 'rain'
        },
        '20200803_151243_part45_1260_1524': {
            'location': 'boulevard', 'time': 'day', 'weather': 'rain'
        },
        '20200706_143808_part26_3660_3860': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200617_191627_part12_1320_1537': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_144800_part25_3610_4360': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_211917_part32_1612_1800': {
            'location': 'highway', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_151313_part23_2632_2808': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_151313_part23_4010_4744': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_143808_part26_500_635': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_195626_part29_1320_1490': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200616_151155_part9_750_900': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_195626_part29_1924_2245': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200618_191030_part17_630_890': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200611_171800_part2_1646_1802': {
            'location': 'boulevard', 'time': 'day', 'weather': 'rain'
        },
        '20200706_161206_part22_2940_3222': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_191736_part30_2212_2515': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_165704_part40_1000_1197': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200616_151155_part9_4020_4306': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200617_191053_part11_18_218': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200610_185206_part1_9850_10050': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_143208_part34_202_467': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200617_190145_part10_930_1269': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200615_184724_part6_5900_6000': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200803_174859_part46_1108_1219': {
            'location': 'highway', 'time': 'day', 'weather': 'rain'
        },
        '20200622_142617_part18_450_910': {
            'location': 'parking_lot', 'time': 'day', 'weather': 'no rain'
        },
        '20200615_171156_part4_7530_7660': {
            'location': 'highway', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_161206_part22_3591_3898': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_191736_part30_1721_1857': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200805_002607_part48_2083_2282': {
            'location': 'boulevard', 'time': 'night', 'weather': 'rain'
        },
        '20200617_191627_part12_1030_1150': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_181359_part42_2671_2829': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_161206_part22_670_950': {
            'location': 'suburban', 'time': 'day', 'weather': 'no rain'
        },
        '20200721_164103_part43_3412_4100': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_171559_part27_10588_11079': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_170136_part28_2688_2884': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200611_184008_part3_2549_2840': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_145605_part24_1484_2248': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_164938_part20_3225_3810': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200805_000536_part47_2225_2325': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_144800_part25_2160_2784': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200730_003948_part44_2995_3195': {
            'location': 'downtown', 'time': 'night', 'weather': 'no rain'
        },
        '20200805_000536_part47_5292_5622': {
            'location': 'downtown', 'time': 'night', 'weather': 'rain'
        },
        '20200721_165008_part39_640_1040': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200616_150451_part8_430_650': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200803_151243_part45_2310_2560': {
            'location': 'boulevard', 'time': 'day', 'weather': 'rain'
        },
        '20200617_190145_part10_2482_2724': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_191736_part30_1860_2209': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200617_195023_part14_1547_1672': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_191736_part30_1211_1322': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200611_171800_part2_942_1152': {
            'location': 'boulevard', 'time': 'day', 'weather': 'rain'
        },
        '20200730_003948_part44_6875_7500': {
            'location': 'downtown', 'time': 'night', 'weather': 'no rain'
        },
        '20200803_151243_part45_1028_1128': {
            'location': 'boulevard', 'time': 'day', 'weather': 'rain'
        },
        '20200618_191030_part17_1120_1509': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200617_192849_part13_2707_2872': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_191736_part30_2731_2869': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200610_185206_part1_5095_5195': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_145605_part24_2450_3046': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200615_184724_part6_5180_5280': {
            'location': 'boulevard', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_143808_part26_1200_1360': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_202209_part31_962_1246': {
            'location': 'suburban', 'time': 'day', 'weather': 'no rain'
        },
        '20200611_184008_part3_1_380': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },
        '20200622_142945_part19_480_700': {
            'location': 'parking_lot', 'time': 'day', 'weather': 'no rain'
        },
        '20200706_162218_part21_2830_3333': {
            'location': 'downtown', 'time': 'day', 'weather': 'no rain'
        },

    }

    dataset_info_counters = {
        'train': {
            'location_counter': {},
            'time_counter': {},
            'weather_counter': {}
        },
        'test': {
            'location_counter': {},
            'time_counter': {},
            'weather_counter': {}
        }
    }

    train_pixset_path = '/home/jfparent/Documents/PixSet/train_dataset/'
    train_dataset_paths = [train_pixset_path+d for d in os.listdir(train_pixset_path)]

    test_pixset_path = '/home/jfparent/Documents/PixSet/test_dataset/'
    test_dataset_paths = [test_pixset_path+d for d in os.listdir(test_pixset_path)]

    dataset_name_splits = ['train', 'test']
    dataset_path_splits = [train_dataset_paths, test_dataset_paths]

    for i_split, dataset_path in enumerate(dataset_path_splits):
        for dataset in dataset_path:
            dataset_name = dataset.split('/')[-1]
            if dataset_info[dataset_name]['location'] in dataset_info_counters[dataset_name_splits[i_split]][
                'location_counter'].keys():
                dataset_info_counters[dataset_name_splits[i_split]]['location_counter'][
                    dataset_info[dataset_name]['location']] += 1
            else:
                dataset_info_counters[dataset_name_splits[i_split]]['location_counter'][
                    dataset_info[dataset_name]['location']] = 1

            if dataset_info[dataset_name]['time'] in dataset_info_counters[dataset_name_splits[i_split]][
                'time_counter'].keys():
                dataset_info_counters[dataset_name_splits[i_split]]['time_counter'][
                    dataset_info[dataset_name]['time']] += 1
            else:
                dataset_info_counters[dataset_name_splits[i_split]]['time_counter'][
                    dataset_info[dataset_name]['time']] = 1

            if dataset_info[dataset_name]['weather'] in dataset_info_counters[dataset_name_splits[i_split]][
                'weather_counter'].keys():
                dataset_info_counters[dataset_name_splits[i_split]]['weather_counter'][
                    dataset_info[dataset_name]['weather']] += 1
            else:
                dataset_info_counters[dataset_name_splits[i_split]]['weather_counter'][
                    dataset_info[dataset_name]['weather']] = 1

    dataset_info_counters['train'] = sort_dict(dataset_info_counters['train'])
    dataset_info_counters['test'] = sort_dict(dataset_info_counters['test'])

    train_location_ratios = list(np.array(list(dataset_info_counters['train']['location_counter'].values())) / np.sum(
        list(dataset_info_counters['train']['location_counter'].values())))
    locations = list(dataset_info_counters['train']['location_counter'].keys())

    test_location_ratios = list(np.array(list(dataset_info_counters['test']['location_counter'].values())) / np.sum(
        list(dataset_info_counters['test']['location_counter'].values())))

    train_time_ratios = list(np.array(list(dataset_info_counters['train']['time_counter'].values())) / np.sum(
        list(dataset_info_counters['train']['time_counter'].values())))
    times = list(dataset_info_counters['train']['time_counter'].keys())

    test_time_ratios = list(np.array(list(dataset_info_counters['test']['time_counter'].values())) / np.sum(
        list(dataset_info_counters['test']['time_counter'].values())))

    train_weather_ratios = list(np.array(list(dataset_info_counters['train']['weather_counter'].values())) / np.sum(
        list(dataset_info_counters['train']['weather_counter'].values())))
    weathers = list(dataset_info_counters['train']['weather_counter'].keys())

    test_weather_ratios = list(np.array(list(dataset_info_counters['test']['weather_counter'].values())) / np.sum(
        list(dataset_info_counters['test']['weather_counter'].values())))

    N = len(locations)
    ind = np.arange(N)
    width = 0.35

    _, ax = plt.subplots()
    train_bars = ax.bar(ind,train_location_ratios, width, color='r')
    test_bars = ax.bar(ind + width, test_location_ratios, width, color='g')

    ax.set_ylabel('Location ratios')
    ax.set_title('Split location ratios')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(locations, rotation=20)

    ax.legend((train_bars[0], test_bars[0]), ('Train', 'Test'))
    plt.savefig('split_location_ratios.png')
    plt.show()

    N = len(times)
    ind = np.arange(N)
    width = 0.35

    _, ax = plt.subplots()
    train_bars = ax.bar(ind,train_time_ratios, width, color='r')
    test_bars = ax.bar(ind + width, test_time_ratios, width, color='g')

    ax.set_ylabel('Time ratios')
    ax.set_title('Split time ratios')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(times, rotation=20)

    ax.legend((train_bars[0], test_bars[0]), ('Train', 'Test'))
    plt.savefig('split_time_ratios.png')
    plt.show()

    N = len(weathers)
    ind = np.arange(N)
    width = 0.35

    _, ax = plt.subplots()
    train_bars = ax.bar(ind,train_weather_ratios, width, color='r')
    test_bars = ax.bar(ind + width, test_weather_ratios, width, color='g')

    ax.set_ylabel('Weather ratios')
    ax.set_title('Split weather ratios')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(weathers, rotation=20)

    ax.legend((train_bars[0], test_bars[0]), ('Train', 'Test'))
    plt.savefig('split_weather_ratios.png')
    plt.show()
