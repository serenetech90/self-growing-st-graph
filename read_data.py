import glob
import math
import pickle

import numpy as np


class DataLoader():

    def __init__(self, args, sel=None , datasets=[0, 1, 2, 3, 4, 5, 6], start=0, processFrame=False, infer=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        parent_dir = '/home/siri0005/Documents/self-growing-spatial-graph/multi_stackLSTM_chains_offline_vel/data'
        # '/Data/stanford_campus_dataset/annotations/'
        # List of data directories where world-coordinates data is stored

        # self.data_dirs = [parent_dir + '/stanford/annotations/bookstore/',
        #                   parent_dir + '/stanford/annotations/hyang/',
        #                   parent_dir + '/stanford/annotations/coupa/',
        #                   parent_dir + '/stanford/annotations/deathCircle/',
        #                   parent_dir + '/stanford/annotations/gates/',
        #                   parent_dir + '/stanford/annotations/nexus/',
        #                   parent_dir + '/crowds/']

        self.data_dirs = [parent_dir + '/stanford/bookstore/',
                          parent_dir + '/stanford/hyang/',
                          parent_dir + '/stanford/coupa/',
                          parent_dir + '/stanford/deathCircle/',
                          parent_dir + '/stanford/gates/',
                          parent_dir + '/stanford/nexus/',
                          parent_dir + '/crowds/test/']

        # parent_dir + '/sdd/pedestrians/quad/',
        # parent_dir + '/sdd/pedestrians/hyang/',
        # parent_dir + '/sdd/pedestrians/coupa/',
        # parent_dir + '/sdd/gates/',
        # parent_dir + '/sdd/little/',
        # parent_dir + '/sdd/deathCircle/'
        # parent_dir + '/eth/',
        # parent_dir + '/hotel/',
        # parent_dir + '/zara/',
        # parent_dir + '/crowds/',
        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.infer = infer

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir = parent_dir

        # Store the arguments
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.pred_len = args.pred_len
        self.diff = self.seq_length

        # Validation arguments
        # self.val_fraction = 0.2

        # Define the path in which the process data would be stored
        self.current_dir = self.used_data_dirs[start]

        files = self.used_data_dirs[start] + "/*.csv" #txt
        # files = self.used_data_dirs[start] + "/video*"
        data_files = sorted(glob.glob(files))

        if sel is None:
            if len(data_files) > 1:
                print([x for x in range(len(data_files))])
                print(data_files)
                sel = input('select which file you want for loading:')

        self.dataset_pointer = str(data_files[int(sel)])[-5] #-1
        if sel < len(data_files):
            self.load_dataset(data_files[int(sel)]) #+'/annotations.txt'

        # If the file doesn't exist or forcePreProcess is true
            if processFrame:
                print("Creating pre-processed data from raw data")
                self.frame_preprocess(self.current_dir + '/trajectories_{0}.cpkl'.format(int(self.dataset_pointer)),
                                      seed=self.seed)

            # Load the processed data from the pickle file
            self.load_trajectories(self.current_dir + 'trajectories_{0}.cpkl'.format(int(self.dataset_pointer)))

    def load_trajectories(self, data_file):
        ''' Load set of pre-processed trajectories from pickled file '''

        f = open(data_file, 'rb')
        self.trajectories = pickle.load(file=f)

        return self.trajectories

    def load_dataset(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file

        f = open(data_file, 'rb')
        self.raw_data = np.genfromtxt(fname=data_file)  # remove, delimiter=','
        f.close()

        if self.raw_data.shape[1] == 10:
            self.frameList = self.raw_data[:, 5]
            ped = self.raw_data[:, 0]
            ind = range(0,int(len(self.raw_data[:, 0])/12))
            x = (self.raw_data[:, 1] + self.raw_data[:, 3])/2
            y = (self.raw_data[:, 2] + self.raw_data[:, 4])/2
            self.pedsPerFrameList = [np.array(ind), ped, x , y]
        # Get all the data from the pickle file
        # self.data = self.raw_data[:,2:4]
        else:
            self.frameList = self.raw_data[:, 0]
            self.pedsPerFrameList = self.raw_data[:, 0:4]
        self.seed = range(len(self.frameList))[0]
        # self.numPedsList = self.raw_data[:,1]
        # self.valid_data = self.raw_data[3]
        # counter = 0
        # valid_counter = 0
        # For each dataset
        # for dataset in range(len(self.data)):
        #     # get the frame data for the current dataset
        #     all_frame_data = self.data[dataset]
        #     valid_frame_data = self.valid_data[dataset]
        #     print('Training data from dataset', dataset, ':', len(all_frame_data))
        #     print('Validation data from dataset', dataset, ':', len(valid_frame_data))
        #     # Increment the counter with the number of sequences in the current dataset
        #     counter += int(len(all_frame_data) / (self.seq_length))
        #     valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # # Calculate the number of batches
        # self.num_batches = int(counter/self.batch_size)
        # self.valid_num_batches = int(valid_counter/self.batch_size)
        # print('Total number of training batches:', self.num_batches * 2)
        # print('Total number of validation batches:', self.valid_num_batches)
        # # On an average, we need twice the number of batches to cover the data
        # # due to randomization introduced
        # self.num_batches = self.num_batches * 2

    def next_step(self, targets):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = {}
        # Target data
        y_batch = []  # np.empty(shape=(1,12), dtype=np.float32)
        # Dataset data
        # d = []
        # Iteration index
        i = 0
        max_idx = max(self.frameList)
        # unique_frames = np.unique(self.frameList)

        max_log = math.log(max_idx, self.seq_length)

        while i < self.batch_size:
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            # treat frames list indices as arithmetic series
            c = (max_idx - (idx + self.seq_length))
            if c <= 0:
                break
            else:
                c = math.log(abs(c), self.diff)
            if c <= max_log:
                # All the data in this sequence
                try:
                    source_frame = self.trajectories[idx]
                except KeyError:
                    self.tick_frame_pointer(valid=False)
                    continue

                # Number of unique peds in this sequence of frames
                x_batch[idx] = source_frame
                idx_c = idx
                for i in range(self.pred_len):
                    targets.append(self.trajectories[idx_c])
                    if idx_c + self.pred_len <= max_idx:
                        idx_c += self.diff  # self.pred_len

                # Number of unique peds in this sequence of frames
                x_batch[idx] = source_frame
                idx_c = idx
                for i in range(self.pred_len):
                    targets.append(self.trajectories[idx_c])
                    if idx_c + self.pred_len <= max_idx:
                        idx_c += self.diff  # self.pred_len

                self.frame_pointer += self.seq_length
            # else:
            #     self.tick_frame_pointer(valid=False, incr=self.seq_length)

            else:
                self.tick_frame_pointer(valid=False, incr=self.seq_length)
            i += 1

        return x_batch, targets, self.frame_pointer  # , d

    def frame_preprocess(self, data_file, seed=0):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        frame_data = {i: {} for i in self.frameList}
        self.frame_pointer = self.seed
        # self.frameList[1] - seed
        j = 0
        x = []
        self.frame_pointer = self.seed
        while self.frame_pointer <= max(self.frameList):
            x = [{ped: [pos_x, pos_y]} for (ind, ped, pos_x, pos_y) in self.pedsPerFrameList
            if ind == self.frame_pointer]
            # for i in range(0,int(max(self.frameList))):
            #     ind = self.pedsPerFrameList[0][i]
            #     ped = self.pedsPerFrameList[1][i]
            #     pos_x = self.pedsPerFrameList[2][i]
            #     pos_y = self.pedsPerFrameList[3][i]
            #     if ind == self.frame_pointer:
            #         x.append({ped: [pos_x, pos_y]})

            frame_data[self.frame_pointer] = x
            # np.select(condlist=[self.pedsPerFrameList[:, 0] == self.frameList[self.frame_pointer]],
            #                    choicelist=[self.pedsPerFrameList[:, 0]], default=-1).sum() np.compress()
            self.tick_frame_pointer(incr=self.diff)

        # while self.frame_pointer <= max(self.frameList):
        #     # This code snippet for biwi_eth frame_ped annotations
        #     for (ind, ped, pos_x, pos_y) in self.pedsPerFrameList:
        #         # if len(frame_data[self.frame_pointer]) and ped in frame_data[self.frame_pointer]:
        #         #     frame_data[self.frame_pointer][ped].append([pos_x, pos_y])
        #         # else:
        #         frame_data[self.frame_pointer]= [{ped: [pos_x, pos_y]}]
        #         self.tick_frame_pointer(incr=self.diff)
        # if ped != self.pedsPerFrameList[int(ind+1)][1]:
        #     self.tick_frame_pointer(incr=self.diff)
        # x = [{ped: [pos_x, pos_y]} for (ind, ped, pos_x, pos_y) in self.pedsPerFrameList
        #      if ind == self.frame_pointer]
        # frame_data[self.frame_pointer] = x
        # np.select(condlist=[self.pedsPerFrameList[:, 0] == self.frameList[self.frame_pointer]],
        #                    choicelist=[self.pedsPerFrameList[:, 0]], default=-1).sum() np.compress()

        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump(frame_data, f, protocol=2)
        f.close()

    def tick_frame_pointer(self, valid=False, incr=12):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            # self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer += incr
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    def reset_data_pointer(self, valid=False, dataset_pointer=0, frame_pointer=0):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = self.seed
        else:
            self.dataset_pointer = dataset_pointer
            self.frame_pointer = frame_pointer
