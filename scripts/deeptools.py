# %% deeptools
import sys
from sys import exit
from deeptools.parserCommon import writableFile, numberOfProcessors
from deeptools import parserCommon
from deeptools import heatmapper
import deeptools.computeMatrixOperations as cmo

parameters = {'upstream': 1000,
            'downstream': 1000,
            'body': 25,
            'bin size': 25,
            'ref point': None,
            'verbose': False,
            'bin avg type': 'mean',
            'missing data as zero': False,
            'min threshold': None,
            'max threshold': None,
            'scale': 1,
            'skip zeros': False,
            'nan after end': False,
            'proc number': 4,
            'sort regions': "descend",
            'sort using': "median",
            'unscaled 5 prime': 0,
            'unscaled 3 prime': 0
            }

hm = heatmapper.heatmapper()

scores_file_list = ['data/%s.25.bw' % chipseq_df['TF_name*'][0]]
hm.computeMatrix(scores_file_list, ["data/hairpins_25.bed"], parameters, blackListFileName=None, verbose=False)

# %%
if parameters['sort regions'] not in ['no', 'keep']:
    sortUsingSamples = []
    # if args.sortUsingSamples is not None:
    #     for i in args.sortUsingSamples:
    #         if (i > 0 and i <= hm.matrix.get_num_samples()):
    #             sortUsingSamples.append(i - 1)
    #         else:
    #             exit("The value {0} for --sortUsingSamples is not valid. Only values from 1 to {1} are allowed.".format(args.sortUsingSamples, hm.matrix.get_num_samples()))
    #     print('Samples used for ordering within each group: ', sortUsingSamples)

    hm.matrix.sort_groups(sort_using=parameters['sort using'], sort_method=parameters['sort regions'], sample_list=sortUsingSamples)
# elif parameters['sort regions'] == 'keep':
#     hm.parameters['group_labels'] = hm.matrix.group_labels
#     hm.parameters["group_boundaries"] = hm.matrix.group_boundaries
#     cmo.sortMatrix(hm, args.regionsFileName, args.transcriptID, args.transcript_id_designator, verbose=not args.quiet)

hm.save_matrix('data/intermediate/%s.mat.gz' % chipseq_df['TF_name*'][0])
